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


# --- experiment 1: does the seed have a signature? ------------------------


@dataclass
class SeedSignature:
    """Result of testing whether a seed value carries a consistent cross-prompt effect."""

    view: str
    model: str
    n_prompts: int
    n_seeds: int
    eta2_seed: float          # share of variance the seed main effect explains
    eta2_prompt: float
    f_stat: float             # MS_seed / MS_residual, residual = prompt x seed
    p_perm: float
    top_features: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "view": self.view, "model": self.model, "n_prompts": self.n_prompts,
            "n_seeds": self.n_seeds, "eta2_seed": self.eta2_seed,
            "eta2_prompt": self.eta2_prompt, "f_stat": self.f_stat,
            "p_perm": self.p_perm, "top_features": self.top_features,
        }


def seed_signature(
    X: np.ndarray,
    prompt_ids: list[str],
    seed_ids: list[int],
    feature_names: list[str] | None = None,
    view: str = "view",
    model: str = "all",
    normalize_rows: bool = False,
    n_perm: int = 999,
    random_state: int = 0,
    top_k: int = 6,
) -> SeedSignature:
    """Test the assumption the main decomposition rests on.

    The decomposition treats seed as a replicate: seed 7 for prompt A is assumed
    to have nothing in common with seed 7 for prompt B. That is an assumption,
    and it is testable — the same seed list is run against every prompt, giving a
    fully crossed prompt x seed grid with one observation per cell.

    With no replication, the prompt x seed interaction is the error term (the
    standard randomised-block layout), so:

        SS_total = SS_prompt + SS_seed + SS_residual
        F = MS_seed / MS_residual

    If the seed effect is real, the seed is not noise — it is a style, applied
    consistently on top of whatever was asked for. That is a much stronger claim
    than "outputs vary", and it makes "uncredited collaborator" literal rather
    than figurative.

    Run this per generative model: seeds mean different things to different
    samplers, and pooling would let a model effect masquerade as a seed effect.
    """
    rng = np.random.default_rng(random_state)
    Z = preprocess(X, normalize_rows=normalize_rows)
    a, prompts = _codes(prompt_ids)
    s, seeds = _codes(seed_ids)
    n_a, n_s = len(prompts), len(seeds)
    if n_a < 2 or n_s < 2:
        raise ValueError("need >= 2 prompts and >= 2 seeds to test for a seed signature")
    if Z.shape[0] != n_a * n_s:
        raise ValueError(
            f"expected a complete {n_a} x {n_s} prompt-by-seed grid "
            f"({n_a * n_s} rows), got {Z.shape[0]}"
        )

    def two_way_no_rep(Z, a, s):
        grand = Z.mean(axis=0)
        am = np.vstack([Z[a == i].mean(axis=0) for i in range(n_a)])
        sm = np.vstack([Z[s == k].mean(axis=0) for k in range(n_s)])
        ss_total = float(((Z - grand) ** 2).sum())
        ss_a = float(n_s * ((am - grand) ** 2).sum())
        ss_s = float(n_a * ((sm - grand) ** 2).sum())
        return ss_total, ss_a, ss_s, ss_total - ss_a - ss_s

    ss_total, ss_a, ss_s, ss_res = two_way_no_rep(Z, a, s)
    df_s, df_res = n_s - 1, (n_a - 1) * (n_s - 1)
    ms_s, ms_res = ss_s / df_s, ss_res / max(df_res, 1)
    f_stat = ms_s / ms_res if ms_res > 0 else float("inf")

    # Under the null the seed labels are exchangeable across prompts, so shuffle
    # them independently within each prompt — that breaks any cross-prompt
    # consistency while leaving each prompt's own spread untouched.
    ge = 0
    for _ in range(n_perm):
        ps = s.copy()
        for i in range(n_a):
            m = a == i
            ps[m] = rng.permutation(ps[m])
        if two_way_no_rep(Z, a, ps)[2] >= ss_s:
            ge += 1
    p_perm = (ge + 1) / (n_perm + 1) if n_perm else float("nan")

    # Which features does the seed act on? Per-dimension seed eta-squared, so the
    # effect can be described ("seed 3 runs consistently darker") rather than
    # only detected.
    top: list[dict] = []
    if feature_names is not None and len(feature_names) == X.shape[1]:
        sd = X.std(axis=0, ddof=0)
        keep = np.flatnonzero(sd > 1e-12)
        names = [feature_names[i] for i in keep]
        grand = Z.mean(axis=0)
        sm = np.vstack([Z[s == k].mean(axis=0) for k in range(n_s)])
        ss_s_dim = n_a * ((sm - grand) ** 2).sum(axis=0)
        ss_t_dim = ((Z - grand) ** 2).sum(axis=0)
        eta_dim = np.divide(ss_s_dim, ss_t_dim, out=np.zeros_like(ss_s_dim), where=ss_t_dim > 0)
        for j in np.argsort(eta_dim)[::-1][:top_k]:
            col = sm[:, j]
            top.append(
                {
                    "feature": names[j],
                    "eta2_seed": float(eta_dim[j]),
                    "strongest_seed": str(seeds[int(np.argmax(np.abs(col)))]),
                    "seed_effect_sd": float(col.std(ddof=0)),
                }
            )

    return SeedSignature(
        view=view,
        model=model,
        n_prompts=n_a,
        n_seeds=n_s,
        eta2_seed=ss_s / ss_total if ss_total else 0.0,
        eta2_prompt=ss_a / ss_total if ss_total else 0.0,
        f_stat=f_stat,
        p_perm=p_perm,
        top_features=top,
    )


# --- experiment 2: the specificity floor ----------------------------------


def rung_curve(
    X: np.ndarray,
    prompt_ids: list[str],
    model_ids: list[str],
    seeds: list[int],
    rungs: list[int],
    families: list[str],
    normalize_rows: bool = False,
    n_boot: int = 500,
    random_state: int = 0,
) -> dict:
    """How much randomness does specification actually remove — and does it hit a floor?

    Seed variance is measured per rung as pooled within-cell variance, where a
    cell is (prompt, model). Features are standardised ONCE across all rungs, so
    the rungs sit in a common space and their variances are directly comparable.

    Note what this deliberately does not do: it never divides by prompt variance.
    A ratio would reintroduce the dependence on which prompts were chosen. The
    claim here is absolute — 'going from 3 words to 35 removes X% of the
    randomness, and the remainder is what specification cannot buy back'.
    """
    rng = np.random.default_rng(random_state)
    Z = preprocess(X, normalize_rows=normalize_rows)
    rungs = [int(r) for r in rungs]
    cells: dict[tuple, list[int]] = {}
    for idx, (p, m) in enumerate(zip(prompt_ids, model_ids)):
        cells.setdefault((p, m), []).append(idx)

    rung_of = {}
    family_of = {}
    for p, r, f in zip(prompt_ids, rungs, families):
        rung_of[p] = r
        family_of[p] = f

    per_rung: dict[int, list[float]] = {}
    per_rung_dist: dict[int, list[float]] = {}
    per_rung_cells: dict[int, list[np.ndarray]] = {}
    for (p, m), idx in cells.items():
        cell = Z[idx]
        if len(cell) < 2:
            continue
        # Mean squared deviation from the cell centroid, per artifact AND per
        # feature dimension. Dividing by the dimension count keeps the number
        # readable and comparable across views of different widths: 1.0 means
        # "as spread out as the whole dataset is".
        var = float(((cell - cell.mean(axis=0)) ** 2).sum() / (len(cell) - 1) / Z.shape[1])
        d = np.linalg.norm(cell[:, None, :] - cell[None, :, :], axis=-1)
        iu = np.triu_indices(len(cell), k=1)
        per_rung.setdefault(rung_of[p], []).append(var)
        per_rung_dist.setdefault(rung_of[p], []).append(float(d[iu].mean()))
        per_rung_cells.setdefault(rung_of[p], []).append(np.asarray(idx))

    ordered = sorted(per_rung)
    base = float(np.mean(per_rung[ordered[0]])) if ordered else float("nan")

    rows = []
    for r in ordered:
        vals = np.array(per_rung[r], dtype=np.float64)
        # Bootstrap over seeds within cell, which is the replication the claim uses.
        boot = []
        for _ in range(n_boot):
            v = []
            for idx in per_rung_cells[r]:
                take = idx[rng.integers(0, len(idx), len(idx))]
                cell = Z[take]
                v.append(
                    float(
                        ((cell - cell.mean(axis=0)) ** 2).sum()
                        / max(len(cell) - 1, 1)
                        / Z.shape[1]
                    )
                )
            boot.append(np.mean(v))
        lo, hi = (
            (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            if n_boot
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "rung": r,
                "n_cells": len(vals),
                "seed_variance": float(vals.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_within_cell_distance": float(np.mean(per_rung_dist[r])),
                "fraction_of_rung1": float(vals.mean() / base) if base else float("nan"),
            }
        )

    removed = 1.0 - rows[-1]["fraction_of_rung1"] if len(rows) > 1 else float("nan")
    return {
        "rows": rows,
        "n_rungs": len(rows),
        "floor_fraction": rows[-1]["fraction_of_rung1"] if rows else float("nan"),
        "randomness_removed": removed,
        "families": sorted(set(families)),
    }


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
