"""Fireworks API client for every call the study makes.

Four endpoints cover both modalities:

  text      POST /inference/v1/chat/completions
  image     POST /inference/v1/image_generation/<model>
  embed     POST /inference/v1/embeddings
  caption   POST /inference/v1/chat/completions  (vision model, data-URI image)

Every response is cached on disk under a content hash of the request, so an
interrupted sweep resumes without re-spending, and re-analysis never re-generates.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

import requests

BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksError(RuntimeError):
    pass


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


class DiskCache:
    """Content-addressed cache. Bytes on disk, key derived from the request."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, kind: str, key: str, ext: str) -> Path:
        d = self.root / kind / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}{ext}"

    def get(self, kind: str, key: str, ext: str) -> bytes | None:
        p = self._path(kind, key, ext)
        return p.read_bytes() if p.exists() else None

    def put(self, kind: str, key: str, ext: str, data: bytes) -> None:
        p = self._path(kind, key, ext)
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        with self._lock:
            tmp.replace(p)


class FireworksClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path = ".cache/fireworks",
        timeout: int = 180,
        max_retries: int = 5,
    ):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise FireworksError(
                "FIREWORKS_API_KEY is not set. Export it, or run with --mock to "
                "exercise the pipeline without spending credits."
            )
        self.cache = DiskCache(cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self._local = threading.local()
        self.calls = 0
        self.cache_hits = 0
        self._counter_lock = threading.Lock()

    @property
    def session(self) -> requests.Session:
        # One session per thread: requests.Session is not documented as thread-safe.
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def _headers(self, accept: str = "application/json") -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": accept,
        }

    def _post(self, url: str, payload: dict, accept: str = "application/json") -> requests.Response:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(
                    url, headers=self._headers(accept), json=payload, timeout=self.timeout
                )
                if r.status_code == 200:
                    with self._counter_lock:
                        self.calls += 1
                    return r
                # 429 and 5xx are worth retrying; a 4xx is our own bug.
                if r.status_code != 429 and r.status_code < 500:
                    raise FireworksError(f"{r.status_code} from {url}: {r.text[:500]}")
                last = FireworksError(f"{r.status_code}: {r.text[:200]}")
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
            sleep = min(2**attempt, 30) + random.random()
            time.sleep(sleep)
        raise FireworksError(f"gave up on {url} after {self.max_retries} attempts: {last}")

    # --- generation ------------------------------------------------------

    def generate_text(
        self,
        model: str,
        prompt: str,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 400,
        system: str | None = None,
    ) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        key = _hash(payload)
        hit = self.cache.get("text", key, ".txt")
        if hit is not None:
            with self._counter_lock:
                self.cache_hits += 1
            return hit.decode()
        r = self._post(f"{BASE_URL}/chat/completions", payload)
        text = r.json()["choices"][0]["message"]["content"]
        self.cache.put("text", key, ".txt", text.encode())
        return text

    def generate_image(
        self,
        model: str,
        prompt: str,
        seed: int,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg_scale: int = 7,
        sampler: str | None = None,
        negative_prompt: str = "",
    ) -> bytes:
        payload: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "samples": 1,
            "safety_check": False,
        }
        if sampler:
            payload["sampler"] = sampler
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        key = _hash({"model": model, **payload})
        hit = self.cache.get("image", key, ".png")
        if hit is not None:
            with self._counter_lock:
                self.cache_hits += 1
            return hit
        r = self._post(f"{BASE_URL}/image_generation/{model}", payload, accept="image/png")
        data = r.content
        if not data or len(data) < 1024:
            raise FireworksError(f"image response too small ({len(data)}B) for seed {seed}")
        self.cache.put("image", key, ".png", data)
        return data

    # --- feature extraction ----------------------------------------------

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed a batch of strings. Cached per string, so batches recombine freely."""
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        keys = [_hash({"model": model, "input": t}) for t in texts]
        for i, k in enumerate(keys):
            hit = self.cache.get("embed", k, ".json")
            if hit is not None:
                out[i] = json.loads(hit)
                with self._counter_lock:
                    self.cache_hits += 1
            else:
                misses.append(i)
        for start in range(0, len(misses), 32):
            chunk = misses[start : start + 32]
            payload = {"model": model, "input": [texts[i] for i in chunk]}
            r = self._post(f"{BASE_URL}/embeddings", payload)
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            for i, item in zip(chunk, data):
                vec = item["embedding"]
                out[i] = vec
                self.cache.put("embed", keys[i], ".json", json.dumps(vec).encode())
        return [v for v in out if v is not None]

    def caption_image(
        self,
        image_bytes: bytes,
        model: str,
        instruction: str = (
            "Describe this image in two or three sentences. Name the main subject, "
            "the setting, the composition, the palette, and the lighting. "
            "Describe only what is visible. Do not speculate or editorialise."
        ),
    ) -> str:
        """A VLM caption is the bridge that puts images into the text embedding space.

        This makes the image and text semantic views commensurable, at the cost of
        whatever the captioner itself smooths over. Captioning runs at temperature 0
        so the caption adds no variance of its own.
        """
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        key = _hash({"model": model, "instruction": instruction, "img": hashlib.sha256(image_bytes).hexdigest()})
        hit = self.cache.get("caption", key, ".txt")
        if hit is not None:
            with self._counter_lock:
                self.cache_hits += 1
            return hit.decode()
        r = self._post(f"{BASE_URL}/chat/completions", payload)
        text = r.json()["choices"][0]["message"]["content"]
        self.cache.put("caption", key, ".txt", text.encode())
        return text


class MockClient:
    """Offline stand-in with a known ground truth, for testing the analysis.

    Outputs are built so that the prompt effect and the seed effect have sizes we
    control (`prompt_strength`, `seed_strength`). If the decomposition can't
    recover those, the decomposition is wrong — which is the point.
    """

    def __init__(self, prompt_strength: float = 1.0, seed_strength: float = 1.0, model_strength: float = 0.3):
        self.prompt_strength = prompt_strength
        self.seed_strength = seed_strength
        self.model_strength = model_strength
        self.calls = 0
        self.cache_hits = 0

    def _rng(self, *parts):
        h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def generate_text(self, model, prompt, seed, **kw):
        p = self._rng("prompt", prompt)
        s = self._rng("seed", prompt, model, seed)
        m = self._rng("model", model)
        words = []
        for i in range(60):
            r = (
                self.prompt_strength * p.gauss(0, 1)
                + self.seed_strength * s.gauss(0, 1)
                + self.model_strength * m.gauss(0, 1)
            )
            words.append(f"w{int(abs(r) * 100) % 500}")
        return " ".join(words) + "."

    def generate_image(self, model, prompt, seed, width=64, height=64, **kw):
        from io import BytesIO

        from PIL import Image

        p = self._rng("prompt", prompt)
        s = self._rng("seed", prompt, model, seed)
        m = self._rng("model", model)

        def channel():
            v = (
                128
                + 40 * self.prompt_strength * p.gauss(0, 1)
                + 40 * self.seed_strength * s.gauss(0, 1)
                + 40 * self.model_strength * m.gauss(0, 1)
            )
            return max(0, min(255, int(v)))

        img = Image.new("RGB", (64, 64))
        px = img.load()
        base = (channel(), channel(), channel())
        for y in range(64):
            for x in range(64):
                jitter = s.randint(-12, 12)
                px[x, y] = tuple(max(0, min(255, c + jitter)) for c in base)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def embed(self, texts, model):
        vecs = []
        for t in texts:
            r = self._rng("embed", t)
            vecs.append([r.gauss(0, 1) for _ in range(64)])
        return vecs

    def caption_image(self, image_bytes, model, instruction=""):
        r = self._rng("cap", hashlib.sha256(image_bytes).hexdigest())
        return " ".join(f"c{r.randint(0, 400)}" for _ in range(20))
