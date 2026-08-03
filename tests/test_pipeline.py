"""End-to-end exercise of sweep -> features -> decomposition -> report, offline."""

from __future__ import annotations

import json

from seedstudy.client import MockClient
from seedstudy.config import StudyConfig
from seedstudy.features import build_views, image_lowlevel, text_lexical
from seedstudy.generate import Manifest, check_balance, run_sweep
from seedstudy.report import write_report
from seedstudy.variance import compare_modalities, decompose


def tiny_config(tmp_path) -> StudyConfig:
    cfg = StudyConfig(name="test", out_dir=str(tmp_path / "run"))
    cfg.seeds = [1, 2, 3, 4]
    cfg.image_models = cfg.image_models[:2]
    cfg.text_models = cfg.text_models[:2]
    cfg.image_prompts = cfg.image_prompts[:3]
    cfg.text_prompts = cfg.text_prompts[:3]
    cfg.image_params.width = cfg.image_params.height = 64
    cfg.caption_images = True
    cfg.max_workers = 4
    return cfg


def test_image_features_are_finite_and_sized():
    v = image_lowlevel(MockClient().generate_image("m", "a landscape", 1))
    assert v.shape == (56,)
    assert (v == v).all()


def test_text_features_capture_length_and_diversity():
    a = text_lexical("The cat sat. The cat sat again.")
    b = text_lexical("Quixotic zephyrs jolted my wavering frame; nevertheless, dawn arrived.")
    assert a[0] < b[0]           # n_chars
    assert a[6] < b[6]           # type-token ratio


def test_sweep_is_balanced_and_resumable(tmp_path):
    cfg = tiny_config(tmp_path)
    client = MockClient()
    m1 = run_sweep(cfg, client)
    expected = sum(cfg.n_cells().values())
    assert len(m1.rows) == expected
    for rep in check_balance(m1).values():
        assert rep["balanced"] and rep["min_cell"] == 4

    # A second sweep must add nothing and must not duplicate manifest rows.
    m2 = run_sweep(cfg, client)
    assert len(m2.rows) == expected
    lines = (cfg.path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == expected


def test_full_pipeline_produces_report(tmp_path):
    cfg = tiny_config(tmp_path)
    client = MockClient()
    manifest = run_sweep(cfg, client)
    views = build_views(
        cfg.path,
        manifest.rows,
        client,
        embed_model=cfg.embed_model,
        vision_model=cfg.vision_model,
        caption_images=True,
        max_workers=2,
    )
    assert set(views) == {"image_lowlevel", "image_semantic", "text_lexical", "text_semantic"}

    decomps = {
        name: decompose(
            v.X, v.prompt_ids, v.model_ids, v.seeds,
            view=name, modality=v.modality, n_perm=49, n_boot=50,
        )
        for name, v in views.items()
    }
    comparison = compare_modalities(decomps)
    paths = write_report(cfg.path, decomps, comparison, {"run": str(cfg.path), "n_artifacts": len(manifest.rows)})

    html = paths["html"].read_text()
    assert "<svg" in html and "Share of output variance" in html
    results = json.loads(paths["json"].read_text())
    assert set(results["decompositions"]) == set(views)
    for d in results["decompositions"].values():
        shares = d["var_share"]
        assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_feature_view_roundtrip(tmp_path):
    cfg = tiny_config(tmp_path)
    client = MockClient()
    manifest = run_sweep(cfg, client, modalities=("text",))
    views = build_views(
        cfg.path, manifest.rows, client,
        embed_model=cfg.embed_model, vision_model=cfg.vision_model, caption_images=False,
    )
    v = views["text_lexical"]
    p = tmp_path / "v.npz"
    v.save(p)
    from seedstudy.features import FeatureView

    back = FeatureView.load(p)
    assert back.name == v.name and back.modality == v.modality
    assert (back.X == v.X).all()
    assert back.prompt_ids == v.prompt_ids and back.seeds == v.seeds


def test_mock_backend_reflects_planted_strengths(tmp_path):
    """Turning the mock's seed strength up must move the measured seed share up."""
    shares = {}
    for label, client in (
        ("prompt_heavy", MockClient(prompt_strength=3.0, seed_strength=0.3, model_strength=0.1)),
        ("seed_heavy", MockClient(prompt_strength=0.3, seed_strength=3.0, model_strength=0.1)),
    ):
        cfg = tiny_config(tmp_path / label)
        cfg.seeds = list(range(1, 9))
        manifest = run_sweep(cfg, client, modalities=("text",))
        views = build_views(
            cfg.path, manifest.rows, client,
            embed_model=cfg.embed_model, vision_model=cfg.vision_model, caption_images=False,
        )
        d = decompose(
            views["text_lexical"].X,
            views["text_lexical"].prompt_ids,
            views["text_lexical"].model_ids,
            views["text_lexical"].seeds,
            n_perm=0, n_boot=0,
        )
        shares[label] = d.var_share["seed"]
    assert shares["seed_heavy"] > shares["prompt_heavy"]


def test_config_roundtrip(tmp_path):
    cfg = tiny_config(tmp_path)
    p = tmp_path / "cfg.json"
    cfg.save(p)
    back = StudyConfig.load(p)
    assert back.seeds == cfg.seeds
    assert back.image_params.width == 64
    assert back.text_params.temperature == cfg.text_params.temperature


def test_manifest_key_dedup(tmp_path):
    m = Manifest(tmp_path / "m.jsonl")
    row = {"modality": "text", "model": "x", "prompt_id": "p", "seed": 1, "path": "a"}
    m.add(row)
    m.add(dict(row))
    assert len(m.rows) == 1
