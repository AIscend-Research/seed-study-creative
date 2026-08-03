"""Experiment 2: can the artist's intent be recovered from the artifact?

Variance shares say how much the output moved. They don't say whether what the
prompt asked for survived into the thing a viewer sees. This asks that directly:
given one output, can you tell which prompt produced it?

A nearest-centroid classifier over the feature space, leave-one-out. Accuracy
above chance is *intent legibility* — the share of the artist's specification
that is actually readable off the artifact. Intent that cannot be recovered from
the work arguably was not exercised on it, which is the sharpest form of the
agency question this study can answer without human subjects.

Classification runs within a generative model, so model identity can't leak in
as a cue. Zero API cost — it runs on features already computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .variance import preprocess


@dataclass
class Legibility:
    view: str
    modality: str
    n_samples: int
    n_classes: int
    accuracy: float
    chance: float
    lift: float                       # (accuracy - chance) / (1 - chance)
    ci_lo: float
    ci_hi: float
    per_rung: list[dict] = field(default_factory=list)
    per_model: list[dict] = field(default_factory=list)
    confusable_pairs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "view": self.view, "modality": self.modality, "n_samples": self.n_samples,
            "n_classes": self.n_classes, "accuracy": self.accuracy, "chance": self.chance,
            "lift": self.lift, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "per_rung": self.per_rung, "per_model": self.per_model,
            "confusable_pairs": self.confusable_pairs,
        }


def _loo_nearest_centroid(Z: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Leave-one-out nearest-centroid predictions, computed in closed form.

    Removing sample i from its own class centroid is a rank-one update, so the
    whole LOO sweep costs one pass rather than n refits. Without this correction
    every sample would be pulled toward its own class and accuracy would be
    optimistic.
    """
    n = Z.shape[0]
    counts = np.array([(y == c).sum() for c in range(n_classes)], dtype=np.float64)
    sums = np.vstack([Z[y == c].sum(axis=0) for c in range(n_classes)])
    centroids = sums / counts[:, None]

    preds = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = y[i]
        cent = centroids.copy()
        if counts[c] > 1:
            cent[c] = (sums[c] - Z[i]) / (counts[c] - 1)
        else:
            cent[c] = np.inf  # a singleton class can't be its own nearest centroid
        d = ((cent - Z[i]) ** 2).sum(axis=1)
        preds[i] = int(np.argmin(d))
    return preds


def intent_legibility(
    X: np.ndarray,
    prompt_ids: list[str],
    model_ids: list[str],
    rungs: list[int] | None = None,
    view: str = "view",
    modality: str = "na",
    normalize_rows: bool = False,
    n_boot: int = 1000,
    random_state: int = 0,
) -> Legibility:
    rng = np.random.default_rng(random_state)
    Z = preprocess(X, normalize_rows=normalize_rows)
    prompts = sorted(set(prompt_ids))
    pindex = {p: i for i, p in enumerate(prompts)}
    y_all = np.array([pindex[p] for p in prompt_ids])
    models = sorted(set(model_ids))

    correct = np.zeros(len(y_all), dtype=bool)
    per_model = []
    confusion = np.zeros((len(prompts), len(prompts)), dtype=np.int64)

    for m in models:
        mask = np.array([mm == m for mm in model_ids])
        Zm, ym = Z[mask], y_all[mask]
        present = sorted(set(ym.tolist()))
        remap = {c: i for i, c in enumerate(present)}
        ym_local = np.array([remap[c] for c in ym])
        preds_local = _loo_nearest_centroid(Zm, ym_local, len(present))
        preds = np.array([present[p] for p in preds_local])
        hit = preds == ym
        correct[mask] = hit
        for t, p in zip(ym, preds):
            confusion[t, p] += 1
        per_model.append(
            {"model": m, "n": int(mask.sum()), "accuracy": float(hit.mean()),
             "chance": 1.0 / len(present)}
        )

    chance = float(np.mean([r["chance"] for r in per_model])) if per_model else float("nan")
    acc = float(correct.mean())
    boot = [
        float(correct[rng.integers(0, len(correct), len(correct))].mean())
        for _ in range(n_boot)
    ] if n_boot else []
    ci = (
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        if boot
        else (float("nan"), float("nan"))
    )

    per_rung = []
    if rungs is not None:
        rr = np.array([int(r) for r in rungs])
        for r in sorted(set(rr.tolist())):
            m = rr == r
            if m.sum():
                per_rung.append(
                    {"rung": int(r), "n": int(m.sum()), "accuracy": float(correct[m].mean())}
                )

    # Which specifications get mistaken for which — the pairs where a viewer
    # could not tell what was asked for.
    pairs = []
    for i in range(len(prompts)):
        for j in range(len(prompts)):
            if i != j and confusion[i, j]:
                pairs.append(
                    {"true": prompts[i], "predicted": prompts[j], "count": int(confusion[i, j])}
                )
    pairs.sort(key=lambda d: -d["count"])

    return Legibility(
        view=view,
        modality=modality,
        n_samples=len(y_all),
        n_classes=len(prompts),
        accuracy=acc,
        chance=chance,
        lift=(acc - chance) / (1 - chance) if chance < 1 else float("nan"),
        ci_lo=ci[0],
        ci_hi=ci[1],
        per_rung=per_rung,
        per_model=per_model,
        confusable_pairs=pairs[:10],
    )
