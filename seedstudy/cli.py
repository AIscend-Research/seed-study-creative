"""Command line entry point.

    python -m seedstudy run --config configs/pilot.json
    python -m seedstudy run --mock                 # no API key, no spend
    python -m seedstudy generate --config ...      # sweep only (resumable)
    python -m seedstudy analyze  --run runs/pilot  # features + ANOVA + report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .client import FireworksClient, MockClient
from .config import StudyConfig, pilot_config
from .features import FeatureView, build_views
from .generate import Manifest, check_balance, run_sweep
from .report import write_report
from .variance import Decomposition, compare_modalities, decompose

# Semantic views live on the unit sphere, so row-normalise before measuring
# distance; the surface views are already interpretable in their own units.
NORMALIZE_ROWS = {"image_semantic": True, "text_semantic": True}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _client(args) -> object:
    if args.mock:
        _log("using MockClient — synthetic outputs, no API calls")
        return MockClient()
    return FireworksClient(cache_dir=args.cache_dir)


def _load_config(args) -> StudyConfig:
    if args.config:
        cfg = StudyConfig.load(args.config)
    elif args.pilot or args.mock:
        cfg = pilot_config(out_dir=args.out or "runs/pilot")
    else:
        cfg = StudyConfig(out_dir=args.out or "runs/default")
    if args.out:
        cfg.out_dir = args.out
    if args.seeds:
        cfg.seeds = list(range(1, args.seeds + 1))
    if args.no_captions:
        cfg.caption_images = False
    if args.workers:
        cfg.max_workers = args.workers
    return cfg


def cmd_generate(args) -> int:
    cfg = _load_config(args)
    cfg.validate()
    counts = cfg.n_cells()
    _log(f"design: {counts['image']} image cells, {counts['text']} text cells -> {cfg.out_dir}")
    client = _client(args)
    modalities = tuple(args.modality) if args.modality else ("image", "text")
    run_sweep(cfg, client, modalities=modalities, progress=_log)
    manifest = Manifest(cfg.path / "manifest.jsonl")
    for modality, rep in check_balance(manifest).items():
        _log(f"{modality}: {rep['n_artifacts']} artifacts, balanced={rep['balanced']} {rep['cell_sizes']}")
    return 0


def cmd_analyze(args) -> int:
    root = Path(args.run)
    cfg = StudyConfig.load(root / "config.json")
    if args.no_captions:
        cfg.caption_images = False
    manifest = Manifest(root / "manifest.jsonl")
    if not manifest.rows:
        _log(f"no artifacts in {root} — run `generate` first")
        return 1

    balance = check_balance(manifest)
    for modality, rep in balance.items():
        _log(f"{modality}: {rep['n_artifacts']} artifacts, balanced={rep['balanced']}")
        if not rep["balanced"]:
            _log(f"  cell sizes {rep['cell_sizes']} — rerun `generate` to fill the gaps")

    client = _client(args)
    feat_dir = root / "features"
    views: dict[str, FeatureView] = {}
    if not args.refresh_features and feat_dir.exists():
        for p in sorted(feat_dir.glob("*.npz")):
            v = FeatureView.load(p)
            views[v.name] = v
        if views:
            _log(f"loaded cached features: {', '.join(sorted(views))}")
    if not views:
        views = build_views(
            root,
            manifest.rows,
            client,
            embed_model=cfg.embed_model,
            vision_model=cfg.vision_model,
            caption_images=cfg.caption_images,
            max_workers=cfg.max_workers,
            progress=_log,
        )
        for name, v in views.items():
            v.save(feat_dir / f"{name}.npz")

    decomps: dict[str, Decomposition] = {}
    for name, v in sorted(views.items()):
        _log(f"decomposing {name}: {v.X.shape[0]} x {v.X.shape[1]}")
        try:
            decomps[name] = decompose(
                v.X,
                v.prompt_ids,
                v.model_ids,
                v.seeds,
                view=name,
                modality=v.modality,
                normalize_rows=NORMALIZE_ROWS.get(name, False),
                n_perm=args.permutations,
                n_boot=args.bootstrap,
                random_state=args.random_state,
            )
        except ValueError as exc:
            _log(f"  skipped {name}: {exc}")

    if not decomps:
        _log("nothing decomposable — check design balance")
        return 1

    comparison = compare_modalities(decomps)
    meta = {
        "run": str(root),
        "study": cfg.name,
        "n_artifacts": len(manifest.rows),
        "seeds": f"{len(cfg.seeds)} per cell",
        "image_models": ", ".join(cfg.image_models),
        "text_models": ", ".join(cfg.text_models),
        "embed_model": cfg.embed_model,
        "vision_model": cfg.vision_model if cfg.caption_images else "(captions off)",
        "permutations": args.permutations,
        "bootstrap": args.bootstrap,
        "balance": json.dumps(balance),
    }
    paths = write_report(root, decomps, comparison, meta)
    for k, p in paths.items():
        _log(f"wrote {k}: {p}")

    print()
    print((root / "report.md").read_text())
    return 0


def cmd_run(args) -> int:
    rc = cmd_generate(args)
    if rc != 0:
        return rc
    cfg = _load_config(args)
    args.run = cfg.out_dir
    return cmd_analyze(args)


def cmd_init(args) -> int:
    cfg = pilot_config() if args.pilot else StudyConfig()
    path = Path(args.output)
    cfg.save(path)
    _log(f"wrote {path} — {cfg.n_cells()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="seedstudy", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="path to a StudyConfig JSON file")
        sp.add_argument("--out", help="output run directory (overrides config)")
        sp.add_argument("--pilot", action="store_true", help="small cheap design")
        sp.add_argument("--mock", action="store_true", help="offline synthetic backend")
        sp.add_argument("--seeds", type=int, help="use seeds 1..N")
        sp.add_argument("--workers", type=int, help="concurrent requests")
        sp.add_argument("--no-captions", action="store_true", help="skip the VLM image captions")
        sp.add_argument("--cache-dir", default=".cache/fireworks")

    g = sub.add_parser("generate", help="run the sweep (resumable)")
    common(g)
    g.add_argument("--modality", nargs="+", choices=["image", "text"])
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("analyze", help="features + variance decomposition + report")
    common(a)
    a.add_argument("--run", required=True, help="run directory produced by `generate`")
    a.add_argument("--permutations", type=int, default=999)
    a.add_argument("--bootstrap", type=int, default=500)
    a.add_argument("--random-state", type=int, default=0)
    a.add_argument("--refresh-features", action="store_true")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("run", help="generate then analyze")
    common(r)
    r.add_argument("--modality", nargs="+", choices=["image", "text"])
    r.add_argument("--permutations", type=int, default=999)
    r.add_argument("--bootstrap", type=int, default=500)
    r.add_argument("--random-state", type=int, default=0)
    r.add_argument("--refresh-features", action="store_true")
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("init", help="write a config file to edit")
    i.add_argument("--output", default="configs/study.json")
    i.add_argument("--pilot", action="store_true")
    i.set_defaults(func=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
