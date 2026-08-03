"""Turn artifacts into feature vectors.

Four views, two per modality, so no conclusion rests on a single way of measuring
"how different are these two outputs":

  image_lowlevel  perceptual statistics computed directly from pixels
  image_semantic  VLM caption -> text embedding  (same space as text_semantic)
  text_semantic   embedding of the generated text
  text_lexical    surface statistics of the generated text

The semantic views are the cross-modal comparison proper: both live in the same
embedding space, so a variance split computed on one is commensurable with the
other. The low-level and lexical views are the cheap, model-free controls.
"""

from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# --- image -------------------------------------------------------------

HUE_BINS = 12
GRID = 4


def image_lowlevel(data: bytes, size: int = 256) -> np.ndarray:
    """Perceptual summary of one image: palette, tone, texture, layout."""
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Hue histogram weighted by saturation: unsaturated pixels have no meaningful hue.
    hue_hist, _ = np.histogram(h.ravel(), bins=HUE_BINS, range=(0, 1), weights=s.ravel())
    hue_hist = hue_hist / (hue_hist.sum() + 1e-9)

    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    # Hasler-Susstrunk colourfulness.
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)

    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)

    # Spatial layout: where the light and the colour sit in the frame.
    cell = size // GRID
    lum_grid = lum.reshape(GRID, cell, GRID, cell).mean(axis=(1, 3)).ravel()
    sat_grid = s.reshape(GRID, cell, GRID, cell).mean(axis=(1, 3)).ravel()

    p = np.histogram(lum.ravel(), bins=32, range=(0, 1))[0].astype(np.float64)
    p /= p.sum() + 1e-9
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    return np.concatenate(
        [
            hue_hist,
            [s.mean(), s.std(), v.mean(), v.std()],
            [lum.mean(), lum.std(), np.percentile(lum, 10), np.percentile(lum, 90)],
            [colorfulness],
            [grad.mean(), grad.std()],
            lum_grid,
            sat_grid,
            [entropy],
        ]
    ).astype(np.float64)


def image_lowlevel_names() -> list[str]:
    names = [f"hue_{i}" for i in range(HUE_BINS)]
    names += ["sat_mean", "sat_std", "val_mean", "val_std"]
    names += ["lum_mean", "lum_std", "lum_p10", "lum_p90"]
    names += ["colorfulness", "grad_mean", "grad_std"]
    names += [f"lum_cell_{i}" for i in range(GRID * GRID)]
    names += [f"sat_cell_{i}" for i in range(GRID * GRID)]
    names += ["lum_entropy"]
    return names


# --- text --------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "but",
    "is", "are", "was", "were", "it", "its", "this", "that", "with", "as",
}


def text_lexical(text: str) -> np.ndarray:
    """Surface statistics of one generation: length, diversity, register, shape."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    n_words = len(words)
    n_chars = len(text)
    sentences = [s for s in SENT_RE.split(text) if s.strip()]
    n_sent = max(len(sentences), 1)
    uniq = set(words)
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    bigrams = set(zip(words, words[1:]))
    lens = np.array([len(w) for w in words], dtype=np.float64) if words else np.array([0.0])

    def rate(pred) -> float:
        return sum(1 for ch in text if pred(ch)) / max(n_chars, 1)

    return np.array(
        [
            n_chars,
            n_words,
            n_sent,
            n_words / n_sent,
            lens.mean(),
            lens.std(),
            len(uniq) / max(n_words, 1),                       # type-token ratio
            hapax / max(n_words, 1),                           # hapax ratio
            len(bigrams) / max(n_words - 1, 1),                # bigram diversity
            sum(1 for w in words if w in FUNCTION_WORDS) / max(n_words, 1),
            rate(str.isdigit),
            rate(str.isupper),
            rate(lambda c: c == ","),
            rate(lambda c: c in ";:—-"),
            text.count("\n\n") + 1,                            # paragraph count
            float(bool(re.search(r"^\s*(?:[-*•]|\d+\.)\s", text, re.M))),  # list markers
        ],
        dtype=np.float64,
    )


def text_lexical_names() -> list[str]:
    return [
        "n_chars", "n_words", "n_sentences", "words_per_sentence",
        "word_len_mean", "word_len_std", "type_token_ratio", "hapax_ratio",
        "bigram_diversity", "function_word_rate", "digit_rate", "upper_rate",
        "comma_rate", "dash_rate", "n_paragraphs", "has_list_markers",
    ]


# --- view assembly -----------------------------------------------------


@dataclass
class FeatureView:
    """A feature matrix plus the design labels for each of its rows."""

    name: str
    modality: str
    X: np.ndarray                # (n_samples, n_features)
    prompt_ids: list[str]
    model_ids: list[str]
    seeds: list[int]
    feature_names: list[str]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            X=self.X,
            prompt_ids=np.array(self.prompt_ids),
            model_ids=np.array(self.model_ids),
            seeds=np.array(self.seeds),
            feature_names=np.array(self.feature_names),
            meta=np.array([json.dumps({"name": self.name, "modality": self.modality})]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureView":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"][0]))
        return cls(
            name=meta["name"],
            modality=meta["modality"],
            X=z["X"],
            prompt_ids=[str(x) for x in z["prompt_ids"]],
            model_ids=[str(x) for x in z["model_ids"]],
            seeds=[int(x) for x in z["seeds"]],
            feature_names=[str(x) for x in z["feature_names"]],
        )


def build_views(
    root: str | Path,
    manifest_rows: list[dict],
    client,
    embed_model: str,
    vision_model: str,
    caption_images: bool = True,
    max_workers: int = 8,
    progress=None,
) -> dict[str, FeatureView]:
    root = Path(root)
    views: dict[str, FeatureView] = {}

    img_rows = sorted(
        [r for r in manifest_rows if r["modality"] == "image"],
        key=lambda r: (r["model"], r["prompt_id"], r["seed"]),
    )
    txt_rows = sorted(
        [r for r in manifest_rows if r["modality"] == "text"],
        key=lambda r: (r["model"], r["prompt_id"], r["seed"]),
    )

    def labels(rows):
        return (
            [r["prompt_id"] for r in rows],
            [r["model"] for r in rows],
            [int(r["seed"]) for r in rows],
        )

    if img_rows:
        if progress:
            progress(f"image: low-level features for {len(img_rows)} artifacts")
        blobs = [(root / r["path"]).read_bytes() for r in img_rows]
        X = np.vstack([image_lowlevel(b) for b in blobs])
        p, m, s = labels(img_rows)
        views["image_lowlevel"] = FeatureView(
            "image_lowlevel", "image", X, p, m, s, image_lowlevel_names()
        )

        if caption_images:
            if progress:
                progress(f"image: captioning {len(img_rows)} artifacts with {vision_model}")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                captions = list(
                    pool.map(lambda b: client.caption_image(b, model=vision_model), blobs)
                )
            (root / "captions.jsonl").write_text(
                "\n".join(
                    json.dumps({**{k: r[k] for k in ("model", "prompt_id", "seed")}, "caption": c})
                    for r, c in zip(img_rows, captions)
                )
            )
            if progress:
                progress("image: embedding captions")
            E = np.asarray(client.embed(captions, model=embed_model), dtype=np.float64)
            views["image_semantic"] = FeatureView(
                "image_semantic", "image", E, p, m, s, [f"e{i}" for i in range(E.shape[1])]
            )

    if txt_rows:
        if progress:
            progress(f"text: lexical features for {len(txt_rows)} artifacts")
        texts = [(root / r["path"]).read_text() for r in txt_rows]
        X = np.vstack([text_lexical(t) for t in texts])
        p, m, s = labels(txt_rows)
        views["text_lexical"] = FeatureView(
            "text_lexical", "text", X, p, m, s, text_lexical_names()
        )
        if progress:
            progress("text: embedding generations")
        E = np.asarray(client.embed(texts, model=embed_model), dtype=np.float64)
        views["text_semantic"] = FeatureView(
            "text_semantic", "text", E, p, m, s, [f"e{i}" for i in range(E.shape[1])]
        )

    return views
