"""The three added experiments, each checked against data with a planted answer."""

from __future__ import annotations

import numpy as np
import pytest

from seedstudy.config import LADDER_IMAGE_PROMPTS, LADDER_TEXT_PROMPTS, ladder_config
from seedstudy.legibility import intent_legibility
from seedstudy.variance import rung_curve, seed_signature

# --- experiment 1: seed signature ---------------------------------------


def crossed(n_prompts=6, n_seeds=20, dims=24, seed_sd=0.0, prompt_sd=1.0, noise_sd=1.0, rs=0):
    """Complete prompt x seed grid, one observation per cell.

    `seed_sd` plants a genuine cross-prompt seed effect: a per-seed offset applied
    identically to every prompt. At 0 the seed is pure noise and the test must not
    find a signature.
    """
    rng = np.random.default_rng(rs)
    p_eff = rng.normal(0, prompt_sd, (n_prompts, dims))
    s_eff = rng.normal(0, seed_sd, (n_seeds, dims)) if seed_sd else np.zeros((n_seeds, dims))
    X, pid, sid = [], [], []
    for i in range(n_prompts):
        for k in range(n_seeds):
            X.append(p_eff[i] + s_eff[k] + rng.normal(0, noise_sd, dims))
            pid.append(f"p{i}")
            sid.append(k)
    return np.vstack(X), pid, sid


def test_detects_a_planted_seed_signature():
    X, p, s = crossed(seed_sd=1.5, noise_sd=0.5)
    sig = seed_signature(X, p, s, n_perm=199)
    assert sig.p_perm < 0.01
    assert sig.f_stat > 2.0


def test_no_signature_when_seed_is_pure_noise():
    X, p, s = crossed(seed_sd=0.0, noise_sd=1.0)
    sig = seed_signature(X, p, s, n_perm=299)
    assert sig.p_perm > 0.05
    assert sig.f_stat == pytest.approx(1.0, abs=0.5)


def test_signature_names_the_features_it_acts_on():
    """Plant the seed effect in two known dimensions; they should surface on top."""
    rng = np.random.default_rng(3)
    dims = 12
    n_p, n_s = 6, 20
    s_eff = np.zeros((n_s, dims))
    s_eff[:, 4] = rng.normal(0, 3.0, n_s)   # only dims 4 and 9 carry a seed effect
    s_eff[:, 9] = rng.normal(0, 3.0, n_s)
    p_eff = rng.normal(0, 1.0, (n_p, dims))
    X, pid, sid = [], [], []
    for i in range(n_p):
        for k in range(n_s):
            X.append(p_eff[i] + s_eff[k] + rng.normal(0, 0.3, dims))
            pid.append(f"p{i}")
            sid.append(k)
    names = [f"f{j}" for j in range(dims)]
    sig = seed_signature(np.vstack(X), pid, sid, feature_names=names, n_perm=99, top_k=3)
    assert {f["feature"] for f in sig.top_features[:2]} == {"f4", "f9"}


def test_signature_requires_a_complete_grid():
    X, p, s = crossed()
    with pytest.raises(ValueError, match="complete"):
        seed_signature(X[:-3], p[:-3], s[:-3], n_perm=0)


# --- experiment 2: intent legibility ------------------------------------


def separable(n_prompts=6, n_seeds=20, dims=16, sep=3.0, rs=0):
    rng = np.random.default_rng(rs)
    cent = rng.normal(0, sep, (n_prompts, dims))
    X, pid, mid = [], [], []
    for i in range(n_prompts):
        for k in range(n_seeds):
            X.append(cent[i] + rng.normal(0, 1.0, dims))
            pid.append(f"p{i}")
            mid.append("m0")
    return np.vstack(X), pid, mid


def test_legibility_high_when_prompts_are_separable():
    X, p, m = separable(sep=5.0)
    leg = intent_legibility(X, p, m, n_boot=200)
    assert leg.accuracy > 0.9
    assert leg.chance == pytest.approx(1 / 6)
    assert leg.lift > 0.85


def test_legibility_at_chance_when_prompt_leaves_no_trace():
    X, p, m = separable(sep=0.0)   # every prompt shares one centroid
    leg = intent_legibility(X, p, m, n_boot=200)
    assert leg.accuracy < 0.35
    assert leg.ci_lo < leg.accuracy < leg.ci_hi


def test_leave_one_out_is_not_optimistic():
    """Without the LOO correction a singleton-ish class trivially predicts itself."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (24, 8))
    p = [f"p{i % 6}" for i in range(24)]
    m = ["m0"] * 24
    leg = intent_legibility(X, p, m, n_boot=0)
    assert leg.accuracy < 0.5   # pure noise must not classify well


def test_legibility_reports_per_rung_when_given_rungs():
    X, p, m = separable(n_prompts=6, sep=4.0)
    rungs = [int(pid[1]) + 1 for pid in p]
    leg = intent_legibility(X, p, m, rungs=rungs, n_boot=0)
    assert [r["rung"] for r in leg.per_rung] == [1, 2, 3, 4, 5, 6]
    assert all(r["n"] == 20 for r in leg.per_rung)


# --- experiment 3: the specificity floor --------------------------------


def ladder_data(seed_sds, n_seeds=15, dims=20, rs=0):
    """One cell per (rung, family), with a controlled seed spread per rung."""
    rng = np.random.default_rng(rs)
    X, pid, mid, sd, rg, fam = [], [], [], [], [], []
    for r, sdv in enumerate(seed_sds, start=1):
        for f in ("landscape", "object"):
            centre = rng.normal(0, 2.0, dims)
            for k in range(n_seeds):
                X.append(centre + rng.normal(0, sdv, dims))
                pid.append(f"{f}_r{r}")
                mid.append("m0")
                sd.append(k)
                rg.append(r)
                fam.append(f)
    return np.vstack(X), pid, mid, sd, rg, fam


def test_floor_curve_is_monotone_when_variance_decays():
    X, p, m, s, rg, fam = ladder_data([2.0, 1.6, 1.2, 0.9, 0.7, 0.6])
    c = rung_curve(X, p, m, s, rg, fam, n_boot=100)
    vals = [r["seed_variance"] for r in sorted(c["rows"], key=lambda x: x["rung"])]
    assert vals == sorted(vals, reverse=True)
    assert c["rows"][0]["fraction_of_rung1"] == pytest.approx(1.0)
    assert 0 < c["floor_fraction"] < 1


def test_floor_detects_a_plateau_above_zero():
    """Decay then flat — the shape the paper's claim depends on."""
    X, p, m, s, rg, fam = ladder_data([2.0, 1.4, 1.0, 1.0, 1.0, 1.0])
    c = rung_curve(X, p, m, s, rg, fam, n_boot=200)
    rows = sorted(c["rows"], key=lambda x: x["rung"])
    tail = [r["seed_variance"] for r in rows[2:]]
    assert max(tail) - min(tail) < 0.25 * rows[0]["seed_variance"]
    assert c["floor_fraction"] > 0.15   # nowhere near zero


def test_floor_reports_full_removal_when_variance_vanishes():
    X, p, m, s, rg, fam = ladder_data([2.0, 1.0, 0.5, 0.2, 0.1, 0.01])
    c = rung_curve(X, p, m, s, rg, fam, n_boot=0)
    assert c["randomness_removed"] > 0.95


def test_bootstrap_band_brackets_each_rung():
    X, p, m, s, rg, fam = ladder_data([2.0, 1.5, 1.2, 1.0, 0.9, 0.8])
    c = rung_curve(X, p, m, s, rg, fam, n_boot=300)
    for r in c["rows"]:
        assert r["ci_lo"] <= r["seed_variance"] <= r["ci_hi"]


# --- the ladder design itself -------------------------------------------


def test_ladder_prompts_are_nested_and_paired():
    img = {p["id"]: p for p in LADDER_IMAGE_PROMPTS}
    txt = {p["id"]: p for p in LADDER_TEXT_PROMPTS}
    assert set(img) == set(txt), "image and text ladders must be paired rung for rung"
    for fam in ("landscape", "object"):
        lens = [len(img[f"{fam}_r{r}"]["text"].split()) for r in range(1, 7)]
        assert lens == sorted(lens), f"{fam} image rungs must not shrink in length"
        lens_t = [len(txt[f"{fam}_r{r}"]["text"].split()) for r in range(1, 7)]
        assert lens_t == sorted(lens_t), f"{fam} text rungs must not shrink in length"


def test_ladder_config_is_balanced_and_sized():
    cfg = ladder_config()
    cfg.validate()
    assert cfg.n_cells() == {"image": 480, "text": 480}
    assert {p["rung"] for p in cfg.image_prompts} == {1, 2, 3, 4, 5, 6}
    assert {p["family"] for p in cfg.image_prompts} == {"landscape", "object"}
