"""The decomposition is only worth trusting if it recovers a variance split we planted."""

from __future__ import annotations

import numpy as np
import pytest

from seedstudy.variance import compare_modalities, decompose


def synth(prompt_sd, seed_sd, model_sd, n_prompts=6, n_models=2, n_seeds=20, dims=32, rs=0):
    """Build data with known variance components, then see if we get them back."""
    rng = np.random.default_rng(rs)
    p_eff = rng.normal(0, prompt_sd, (n_prompts, dims))
    m_eff = rng.normal(0, model_sd, (n_models, dims))
    X, pid, mid, seeds = [], [], [], []
    for i in range(n_prompts):
        for j in range(n_models):
            for s in range(n_seeds):
                X.append(p_eff[i] + m_eff[j] + rng.normal(0, seed_sd, dims))
                pid.append(f"p{i}")
                mid.append(f"m{j}")
                seeds.append(s)
    return np.vstack(X), pid, mid, seeds


def test_recovers_prompt_dominant_split():
    X, p, m, s = synth(prompt_sd=2.0, seed_sd=0.5, model_sd=0.2)
    d = decompose(X, p, m, s, n_perm=99, n_boot=100)
    assert d.var_share["prompt"] > d.var_share["seed"]
    assert d.seed_to_prompt < 1.0
    # planted ratio of variances is 0.25/4 = 0.0625
    assert 0.02 < d.seed_to_prompt < 0.2
    assert d.get("prompt")["p_perm"] < 0.05


def test_recovers_seed_dominant_split():
    X, p, m, s = synth(prompt_sd=0.4, seed_sd=2.0, model_sd=0.2)
    d = decompose(X, p, m, s, n_perm=99, n_boot=100)
    assert d.var_share["seed"] > d.var_share["prompt"]
    assert d.seed_to_prompt > 1.0
    assert d.p_seed_beats_prompt > 0.4  # prompt barely separates anything


def test_pure_noise_has_no_prompt_effect():
    X, p, m, s = synth(prompt_sd=1e-9, seed_sd=1.0, model_sd=1e-9)
    d = decompose(X, p, m, s, n_perm=199, n_boot=0)
    assert d.var_share["seed"] > 0.95
    assert d.get("prompt")["p_perm"] > 0.05


def test_sums_of_squares_are_orthogonal():
    X, p, m, s = synth(1.0, 1.0, 1.0)
    d = decompose(X, p, m, s, n_perm=0, n_boot=0)
    total_eta = sum(r["eta2"] for r in d.table)
    assert total_eta == pytest.approx(1.0, abs=1e-9)


def test_degrees_of_freedom():
    X, p, m, s = synth(1.0, 1.0, 1.0, n_prompts=6, n_models=2, n_seeds=20)
    d = decompose(X, p, m, s, n_perm=0, n_boot=0)
    got = {r["source"]: r["df"] for r in d.table}
    assert got == {"prompt": 5, "model": 1, "prompt:model": 5, "seed": 6 * 2 * 19}


def test_single_model_falls_back_to_one_way():
    X, p, m, s = synth(1.0, 1.0, 0.0, n_models=1)
    d = decompose(X, p, m, s, n_perm=0, n_boot=0)
    assert d.var_share["model"] == 0.0
    assert d.var_share["prompt:model"] == 0.0
    assert d.var_share["prompt"] + d.var_share["seed"] == pytest.approx(1.0)


def test_unbalanced_design_is_refused():
    X, p, m, s = synth(1.0, 1.0, 1.0, n_seeds=5)
    with pytest.raises(ValueError, match="unbalanced"):
        decompose(X[:-1], p[:-1], m[:-1], s[:-1], n_perm=0, n_boot=0)


def test_bootstrap_interval_brackets_the_estimate():
    X, p, m, s = synth(prompt_sd=1.0, seed_sd=1.0, model_sd=0.3)
    d = decompose(X, p, m, s, n_perm=0, n_boot=300)
    lo, hi = d.seed_to_prompt_ci
    assert lo < d.seed_to_prompt < hi


def test_compare_modalities_pairs_views():
    X, p, m, s = synth(1.0, 1.0, 0.3)
    a = decompose(X, p, m, s, view="image_semantic", modality="image", n_perm=0, n_boot=0)
    b = decompose(X, p, m, s, view="text_semantic", modality="text", n_perm=0, n_boot=0)
    cmp_ = compare_modalities({"image_semantic": a, "text_semantic": b})
    assert cmp_["semantic_pair"]["seed_share_gap"] == pytest.approx(0.0, abs=1e-9)
    assert len(cmp_["per_view"]) == 2
