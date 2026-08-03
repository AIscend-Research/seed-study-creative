"""Decompose output variance into seed, prompt, and model.

The design is prompt x model fully crossed with `n` seeds per cell. Seed is not a
factor whose levels mean anything across cells — seed 7 for prompt A has no
relationship to seed 7 for prompt B — so it is not a main effect. It is the
*replicate* dimension, and because the seed is the only thing varying inside a
cell, within-cell variance IS seed variance. That is the whole design:

    SS_total = SS_prompt + SS_model + SS_prompt:model + SS_within
                                                        ^^^^^^^^^ the seed

Outputs are vectors, so every sum of squares is a sum over feature dimensions
(equivalently the trace of the Gower-centred cross-product matrix — with
Euclidean distances the two are identical). Because the design is balanced and
fully crossed, the decomposition is exactly orthogonal and the components are
exactly estimable, with no ordering ambiguity.

Two summaries are reported, and they answer different questions:

  eta-squared          share of the variance in THIS SAMPLE, per source.
                       Inflated for prompt/model, because their mean squares
                       carry seed noise (E[MS_A] > n*b*sigma2_A).
  variance components  the EMS-corrected estimate of the underlying population
                       variance per source. This is the honest apples-to-apples
                       number and it is what the headline claim rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def preprocess(X: np.ndarray, normalize_rows: bool = False) -> np.ndarray:
    """Put features on a common scale so no single dimension dominates the SS.

    `normalize_rows` first projects each row onto the unit sphere, which makes
    squared Euclidean distance a monotone function of cosine distance — the right
    geometry for embeddings.
    """
    X = np.asarray(X, dtype=np.float64)
    if normalize_rows:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms == 0, 1.0, norms)
    sd = X.std(axis=0, ddof=0)
    keep = sd > 1e-12
    if not keep.any():
        raise ValueError("all features are constant — nothing to decompose")
    X = X[:, keep]
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)


def _codes(labels) -> tuple[np.ndarray, list]:
    levels = sorted(set(labels))
    index = {lv: i for i, lv in enumerate(levels)}
    return np.array([index[x] for x in labels]), levels


def _ss(X: np.ndarray, a: np.ndarray, b: np.ndarray, n_a: int, n_b: int) -> dict:
    """Balanced two-way sums of squares, summed over feature dimensions."""
    grand = X.mean(axis=0)
    ss_total = float(((X - grand) ** 2).sum())

    a_means = np.vstack([X[a == i].mean(axis=0) for i in range(n_a)])
    b_means = np.vstack([X[b == j].mean(axis=0) for j in range(n_b)])
    n_per_a = np.array([(a == i).sum() for i in range(n_a)])
    n_per_b = np.array([(b == j).sum() for j in range(n_b)])

    ss_a = float((n_per_a[:, None] * (a_means - grand) ** 2).sum())
    ss_b = float((n_per_b[:, None] * (b_means - grand) ** 2).sum())

    ss_ab = 0.0
    ss_within = 0.0
    for i in range(n_a):
        for j in range(n_b):
            mask = (a == i) & (b == j)
            cell = X[mask]
            cm = cell.mean(axis=0)
            ss_within += float(((cell - cm) ** 2).sum())
            ss_ab += float(mask.sum()) * float(
                ((cm - a_means[i] - b_means[j] + grand) ** 2).sum()
            )
    return {"total": ss_total, "a": ss_a, "b": ss_b, "ab": ss_ab, "within": ss_within}


@dataclass
class Decomposition:
    view: str
    modality: str
    n_samples: int
    n_prompts: int
    n_models: int
    n_seeds: int
    n_features: int
    table: list[dict] = field(default_factory=list)     # per source: df, SS, MS, eta2, var comp
    var_share: dict = field(default_factory=dict)       # source -> share of total variance
    seed_to_prompt: float = float("nan")
    seed_to_prompt_ci: tuple[float, float] = (float("nan"), float("nan"))
    p_seed_beats_prompt: float = float("nan")
    mean_within_prompt_dist: float = float("nan")
    mean_between_prompt_dist: float = float("nan")

    def get(self, source: str) -> dict:
        for row in self.table:
            if row["source"] == source:
                return row
        raise KeyError(source)

    def to_dict(self) -> dict:
        return {
            "view": self.view,
            "modality": self.modality,
            "n_samples": self.n_samples,
            "n_prompts": self.n_prompts,
            "n_models": self.n_models,
            "n_seeds": self.n_seeds,
            "n_features": self.n_features,
            "table": self.table,
            "var_share": self.var_share,
            "seed_to_prompt": self.seed_to_prompt,
            "seed_to_prompt_ci": list(self.seed_to_prompt_ci),
            "p_seed_beats_prompt": self.p_seed_beats_prompt,
            "mean_within_prompt_dist": self.mean_within_prompt_dist,
            "mean_between_prompt_dist": self.mean_between_prompt_dist,
        }


def _components(ms: dict, n: int, n_a: int, n_b: int) -> dict:
    """EMS-corrected variance components, both factors treated as random.

    For a balanced two-factor random model with n replicates:
        E[MS_A]  = s2_e + n*s2_AB + n*b*s2_A
        E[MS_B]  = s2_e + n*s2_AB + n*a*s2_B
        E[MS_AB] = s2_e + n*s2_AB
        E[MS_E]  = s2_e
    Treating prompt and model as random is the right call here: we care about the
    variance a *typical* prompt or model induces, not about these six prompts.
    Components are clamped at zero — the usual ANOVA estimator can go negative
    when the true component is near zero.
    """
    s2_e = ms["within"]
    if n_b > 1:
        s2_ab = max(0.0, (ms["ab"] - ms["within"]) / n)
        s2_a = max(0.0, (ms["a"] - ms["ab"]) / (n * n_b))
        s2_b = max(0.0, (ms["b"] - ms["ab"]) / (n * n_a))
    else:
        s2_ab = 0.0
        s2_a = max(0.0, (ms["a"] - ms["within"]) / n)
        s2_b = 0.0
    return {"prompt": s2_a, "model": s2_b, "prompt:model": s2_ab, "seed": s2_e}


def _pair_stats(X: np.ndarray, a: np.ndarray, rng: np.random.Generator, max_pairs: int = 200_000):
    """Interpretable distances: two seeds of one prompt vs two different prompts.

    P(seed pair > prompt pair) is the probability that two outputs sharing a
    prompt are farther apart than two outputs from different prompts. At 0.5 the
    prompt carries no usable signal at all; below 0.5 it separates outputs.
    """
    n = X.shape[0]
    # Sampling rather than the full n^2 matrix keeps this linear at large n.
    reps = min(max_pairs, n * 40)
    i = rng.integers(0, n, reps)
    j = rng.integers(0, n, reps)
    ok = i != j
    i, j = i[ok], j[ok]
    d = np.linalg.norm(X[i] - X[j], axis=1)
    same = a[i] == a[j]
    same_d = d[same]
    diff_d = d[~same]
    if len(same_d) == 0 or len(diff_d) == 0:
        return float("nan"), float("nan"), float("nan")
    k = min(len(same_d), len(diff_d), 50_000)
    s = rng.choice(same_d, k, replace=False)
    t = rng.choice(diff_d, k, replace=False)
    p_beats = float((s > t).mean() + 0.5 * (s == t).mean())
    return float(same_d.mean()), float(diff_d.mean()), p_beats


def decompose(
    X: np.ndarray,
    prompt_ids: list[str],
    model_ids: list[str],
    seeds: list[int],
    view: str = "view",
    modality: str = "na",
    normalize_rows: bool = False,
    n_perm: int = 999,
    n_boot: int = 500,
    random_state: int = 0,
) -> Decomposition:
    rng = np.random.default_rng(random_state)
    Z = preprocess(X, normalize_rows=normalize_rows)
    a, prompts = _codes(prompt_ids)
    b, models = _codes(model_ids)
    n_a, n_b = len(prompts), len(models)
    N = Z.shape[0]

    cell_counts = {(i, j): int(((a == i) & (b == j)).sum()) for i in range(n_a) for j in range(n_b)}
    sizes = set(cell_counts.values())
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError(
            f"unbalanced design for view {view}: cell sizes {sorted(sizes)}. "
            "Rerun `generate` to fill missing cells, or drop the incomplete ones."
        )
    n = sizes.pop()
    if n < 2:
        raise ValueError("need >= 2 seeds per cell to separate seed variance from the mean")

    ss = _ss(Z, a, b, n_a, n_b)
    df = {
        "prompt": n_a - 1,
        "model": max(n_b - 1, 0),
        "prompt:model": (n_a - 1) * (n_b - 1),
        "seed": n_a * n_b * (n - 1),
    }
    ss_named = {"prompt": ss["a"], "model": ss["b"], "prompt:model": ss["ab"], "seed": ss["within"]}
    ms = {k: (ss_named[k] / df[k] if df[k] > 0 else 0.0) for k in ss_named}
    comps = _components(
        {"a": ms["prompt"], "b": ms["model"], "ab": ms["prompt:model"], "within": ms["seed"]},
        n, n_a, n_b,
    )
    comp_total = sum(comps.values()) or 1.0

    pvals = _permutation_pvalues(Z, a, b, n_a, n_b, ss, n_perm, rng) if n_perm else {}

    table = []
    for source in ("prompt", "model", "prompt:model", "seed"):
        table.append(
            {
                "source": source,
                "df": df[source],
                "ss": ss_named[source],
                "ms": ms[source],
                "eta2": ss_named[source] / ss["total"] if ss["total"] else 0.0,
                "var_component": comps[source],
                "var_share": comps[source] / comp_total,
                "p_perm": pvals.get(source, float("nan")),
            }
        )

    ratio = comps["seed"] / comps["prompt"] if comps["prompt"] > 0 else float("inf")
    ci = (
        _bootstrap_ratio(Z, a, b, n_a, n_b, n, n_boot, rng, ratio)
        if n_boot
        else (float("nan"), float("nan"))
    )
    win, betw, p_beats = _pair_stats(Z, a, rng)

    return Decomposition(
        view=view,
        modality=modality,
        n_samples=N,
        n_prompts=n_a,
        n_models=n_b,
        n_seeds=n,
        n_features=Z.shape[1],
        table=table,
        var_share={r["source"]: r["var_share"] for r in table},
        seed_to_prompt=ratio,
        seed_to_prompt_ci=ci,
        p_seed_beats_prompt=p_beats,
        mean_within_prompt_dist=win,
        mean_between_prompt_dist=betw,
    )


def _permutation_pvalues(Z, a, b, n_a, n_b, observed, n_perm, rng) -> dict:
    """Label-shuffling tests, each restricted so it only breaks the effect tested.

    Prompt labels are shuffled within model (and vice versa) so the other factor's
    structure survives the null; the interaction is tested by freely permuting the
    residuals of the additive model, which is the effect the interaction adds.
    """
    out = {}

    ge_a = ge_b = 0
    for _ in range(n_perm):
        pa = a.copy()
        for j in range(n_b):
            m = b == j
            pa[m] = rng.permutation(pa[m])
        if _ss(Z, pa, b, n_a, n_b)["a"] >= observed["a"]:
            ge_a += 1
        if n_b > 1:
            pb = b.copy()
            for i in range(n_a):
                m = a == i
                pb[m] = rng.permutation(pb[m])
            if _ss(Z, a, pb, n_a, n_b)["b"] >= observed["b"]:
                ge_b += 1
    out["prompt"] = (ge_a + 1) / (n_perm + 1)
    out["model"] = (ge_b + 1) / (n_perm + 1) if n_b > 1 else float("nan")

    if n_b > 1:
        grand = Z.mean(axis=0)
        am = np.vstack([Z[a == i].mean(axis=0) for i in range(n_a)])
        bm = np.vstack([Z[b == j].mean(axis=0) for j in range(n_b)])
        fitted = am[a] + bm[b] - grand
        resid = Z - fitted
        ge_ab = 0
        for _ in range(n_perm):
            perm = rng.permutation(len(resid))
            if _ss(fitted + resid[perm], a, b, n_a, n_b)["ab"] >= observed["ab"]:
                ge_ab += 1
        out["prompt:model"] = (ge_ab + 1) / (n_perm + 1)
    else:
        out["prompt:model"] = float("nan")

    # Seed is the residual stratum: there is no label to shuffle, and its variance
    # is bounded away from zero by construction whenever any two seeds differ.
    out["seed"] = float("nan")
    return out


def _bootstrap_ratio(Z, a, b, n_a, n_b, n, n_boot, rng, observed_ratio) -> tuple[float, float]:
    """Two-level bootstrap over prompts and over seeds within cell.

    Resampling prompts is what makes the interval honest: with six prompts, the
    dominant uncertainty in "how big is the prompt effect" is which prompts we
    happened to pick, not how many seeds we ran.

    Resampling prompts with replacement also biases the ratio upward — duplicated
    prompts share a mean, which shrinks the between-prompt sum of squares. So the
    interval is the *basic* (reverse-percentile) bootstrap taken on the log scale:
    it reflects the resampling spread back around the observed estimate instead of
    inheriting that shift, and the log keeps a ratio bounded below by zero.
    """
    cell_idx = {
        (i, j): np.flatnonzero((a == i) & (b == j)) for i in range(n_a) for j in range(n_b)
    }
    ratios = []
    for _ in range(n_boot):
        prompt_pick = rng.integers(0, n_a, n_a)
        rows, ra, rb = [], [], []
        for new_i, i in enumerate(prompt_pick):
            for j in range(n_b):
                idx = cell_idx[(i, j)]
                take = idx[rng.integers(0, n, n)]
                rows.append(take)
                ra.append(np.full(n, new_i))
                rb.append(np.full(n, j))
        rows = np.concatenate(rows)
        ra = np.concatenate(ra)
        rb = np.concatenate(rb)
        # Resampling with replacement can collapse a prompt's spread; guard against
        # the degenerate draw rather than letting it produce a spurious infinity.
        try:
            s = _ss(Z[rows], ra, rb, n_a, n_b)
        except Exception:
            continue
        dfm = {
            "a": n_a - 1,
            "b": max(n_b - 1, 0),
            "ab": (n_a - 1) * (n_b - 1),
            "within": n_a * n_b * (n - 1),
        }
        ms = {k: (s[k] / dfm[k] if dfm[k] > 0 else 0.0) for k in ("a", "b", "ab", "within")}
        c = _components(ms, n, n_a, n_b)
        if c["prompt"] > 0:
            ratios.append(c["seed"] / c["prompt"])
        elif c["seed"] > 0:
            ratios.append(np.inf)
    arr = np.array(ratios, dtype=np.float64)
    usable = arr[np.isfinite(arr) & (arr > 0)]
    if usable.size < 20:
        return (float("nan"), float("nan"))
    # Draws where the prompt component collapsed to zero give an infinite ratio;
    # too many of those and the interval has no finite upper bound to report.
    inf_frac = 1.0 - usable.size / arr.size

    if not np.isfinite(observed_ratio) or observed_ratio <= 0:
        lo = float(np.percentile(usable, 2.5))
        hi = float("inf") if inf_frac > 0.025 else float(np.percentile(usable, 97.5))
        return (lo, hi)

    logs = np.log(usable)
    q_lo, q_hi = np.percentile(logs, [2.5, 97.5])
    centre = np.log(observed_ratio)
    lo = float(np.exp(2 * centre - q_hi))
    hi = float("inf") if inf_frac > 0.025 else float(np.exp(2 * centre - q_lo))
    return (lo, hi)


def compare_modalities(decomps: dict[str, Decomposition]) -> dict:
    """Line the modalities up on the shares, which are scale-free and comparable.

    The semantic pair is the load-bearing comparison — image captions and text
    generations are embedded by the same model into the same space, so their
    variance shares mean the same thing. The low-level/lexical pair is a control:
    agreement there means the result is not an artifact of the embedder.
    """
    rows = []
    for name, d in sorted(decomps.items()):
        rows.append(
            {
                "view": name,
                "modality": d.modality,
                "seed_share": d.var_share.get("seed", float("nan")),
                "prompt_share": d.var_share.get("prompt", float("nan")),
                "model_share": d.var_share.get("model", float("nan")),
                "interaction_share": d.var_share.get("prompt:model", float("nan")),
                "seed_to_prompt": d.seed_to_prompt,
                "seed_to_prompt_ci": list(d.seed_to_prompt_ci),
                "p_seed_beats_prompt": d.p_seed_beats_prompt,
            }
        )

    def pair(img_view, txt_view):
        if img_view not in decomps or txt_view not in decomps:
            return None
        i, t = decomps[img_view], decomps[txt_view]
        return {
            "image_view": img_view,
            "text_view": txt_view,
            "image_seed_share": i.var_share["seed"],
            "text_seed_share": t.var_share["seed"],
            "seed_share_gap": i.var_share["seed"] - t.var_share["seed"],
            "image_seed_to_prompt": i.seed_to_prompt,
            "text_seed_to_prompt": t.seed_to_prompt,
        }

    return {
        "per_view": rows,
        "semantic_pair": pair("image_semantic", "text_semantic"),
        "surface_pair": pair("image_lowlevel", "text_lexical"),
    }
