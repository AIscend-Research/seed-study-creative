"""Study design: what gets held fixed, what gets swept.

The design is a fully crossed prompt x model x seed grid, run once per cell.
Balance matters: the variance decomposition in `variance.py` assumes every
(prompt, model) cell has the same number of seeds, which makes the sums of
squares exactly orthogonal and the components exactly estimable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Fireworks model ids. Image models are SDXL-class; text models are open-weight
# instruct models that honour the `seed` parameter on /chat/completions.
DEFAULT_IMAGE_MODELS = [
    "accounts/fireworks/models/stable-diffusion-xl-1024-v1-0",
    "accounts/fireworks/models/playground-v2-1024px-aesthetic",
]
DEFAULT_TEXT_MODELS = [
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/qwen2p5-7b-instruct",
]
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_VISION_MODEL = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"

# Prompts are matched in pairs across modalities: the same scene brief, phrased
# for a diffusion model and for a chat model. Specificity is deliberately varied
# (loose -> tight) because the prompt/seed balance is expected to shift with it.
DEFAULT_IMAGE_PROMPTS = [
    {"id": "loose_landscape", "text": "a landscape", "specificity": "loose"},
    {"id": "loose_portrait", "text": "a portrait of a person", "specificity": "loose"},
    {
        "id": "mid_lighthouse",
        "text": "a lighthouse on a rocky coast at dusk",
        "specificity": "mid",
    },
    {
        "id": "mid_market",
        "text": "a busy street market in the rain, neon signs",
        "specificity": "mid",
    },
    {
        "id": "tight_teapot",
        "text": (
            "a single blue ceramic teapot centred on a white seamless backdrop, "
            "soft frontal studio lighting, no other objects, product photograph, "
            "sharp focus, eye-level camera"
        ),
        "specificity": "tight",
    },
    {
        "id": "tight_diagram",
        "text": (
            "a flat vector diagram of a red circle directly above a black square "
            "on a plain white background, thick uniform outlines, no text, no shading"
        ),
        "specificity": "tight",
    },
]

DEFAULT_TEXT_PROMPTS = [
    {"id": "loose_landscape", "text": "Describe a landscape.", "specificity": "loose"},
    {"id": "loose_portrait", "text": "Describe a person.", "specificity": "loose"},
    {
        "id": "mid_lighthouse",
        "text": "Write a short paragraph about a lighthouse on a rocky coast at dusk.",
        "specificity": "mid",
    },
    {
        "id": "mid_market",
        "text": "Write a short paragraph about a busy street market in the rain.",
        "specificity": "mid",
    },
    {
        "id": "tight_teapot",
        "text": (
            "Write exactly three sentences describing a single blue ceramic teapot "
            "on a white background. Mention only the teapot. Use plain declarative "
            "sentences in the present tense. Do not add a title or any preamble."
        ),
        "specificity": "tight",
    },
    {
        "id": "tight_diagram",
        "text": (
            "Describe, in exactly two sentences, a diagram showing a red circle "
            "directly above a black square on a white background. State the shapes, "
            "their colours, and their relative positions and nothing else."
        ),
        "specificity": "tight",
    },
]


@dataclass
class ImageParams:
    """Generation settings held constant across the whole image sweep."""

    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: int = 7
    sampler: str | None = None
    negative_prompt: str = ""


@dataclass
class TextParams:
    """Generation settings held constant across the whole text sweep.

    Temperature must be > 0 or the seed has nothing to act on: greedy decoding
    is deterministic regardless of seed, and every "seed variance" would be zero
    by construction rather than by measurement.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 400
    system: str | None = None


@dataclass
class StudyConfig:
    name: str = "seed-study"
    out_dir: str = "runs/default"

    seeds: list[int] = field(default_factory=lambda: list(range(1, 21)))

    image_models: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_MODELS))
    text_models: list[str] = field(default_factory=lambda: list(DEFAULT_TEXT_MODELS))
    image_prompts: list[dict] = field(
        default_factory=lambda: [dict(p) for p in DEFAULT_IMAGE_PROMPTS]
    )
    text_prompts: list[dict] = field(
        default_factory=lambda: [dict(p) for p in DEFAULT_TEXT_PROMPTS]
    )

    image_params: ImageParams = field(default_factory=ImageParams)
    text_params: TextParams = field(default_factory=TextParams)

    embed_model: str = DEFAULT_EMBED_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    # Captioning every image costs a VLM call per artifact; low-level image
    # features are free. Semantic image features are the more interesting view,
    # so they are on by default and can be switched off for a cheap pilot.
    caption_images: bool = True

    max_workers: int = 8

    @property
    def path(self) -> Path:
        return Path(self.out_dir)

    def n_cells(self) -> dict[str, int]:
        return {
            "image": len(self.image_models) * len(self.image_prompts) * len(self.seeds),
            "text": len(self.text_models) * len(self.text_prompts) * len(self.seeds),
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "StudyConfig":
        raw = json.loads(Path(path).read_text())
        img = ImageParams(**raw.pop("image_params", {}))
        txt = TextParams(**raw.pop("text_params", {}))
        return cls(image_params=img, text_params=txt, **raw)

    def validate(self) -> None:
        if len(self.seeds) < 3:
            raise ValueError("need >= 3 seeds per cell to estimate within-cell variance")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds would break the balanced design")
        for label, prompts in (("image", self.image_prompts), ("text", self.text_prompts)):
            if len(prompts) < 2:
                raise ValueError(f"need >= 2 {label} prompts to estimate a prompt effect")
            ids = [p["id"] for p in prompts]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {label} prompt ids")


def pilot_config(out_dir: str = "runs/pilot") -> StudyConfig:
    """A small, cheap design for smoke-testing the whole pipeline end to end."""
    cfg = StudyConfig(name="pilot", out_dir=out_dir)
    cfg.seeds = list(range(1, 6))
    cfg.image_models = cfg.image_models[:1]
    cfg.text_models = cfg.text_models[:1]
    cfg.image_prompts = [p for p in cfg.image_prompts if p["id"].startswith(("loose", "tight"))][:2]
    cfg.text_prompts = [p for p in cfg.text_prompts if p["id"].startswith(("loose", "tight"))][:2]
    cfg.image_params.steps = 20
    cfg.text_params.max_tokens = 200
    return cfg
