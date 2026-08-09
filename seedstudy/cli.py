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
from .config import StudyConfig, ladder_config, pilot_config
from .features import FeatureView, build_views
from .generate import Manifest, check_balance, run_sweep, slug
from .report import write_report
from .legibility import intent_legibility
from .variance import (
    Decomposition,
    compare_modalities,
    decompose,
    rung_curve,
    seed_signature,
)

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
    elif getattr(args, "ladder", False):
        cfg = ladder_config(out_dir=args.out or "runs/ladder")
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

    # Experiment 1: is the seed a style rather than noise? Run per generative
    # model — seeds mean different things to different samplers, and pooling
    # would let a model effect pose as a seed effect.
    signatures: dict[str, list] = {}
    for name, v in sorted(views.items()):
        # The test asks whether seed 7 means the same thing across prompts. That
        # is only a question if the backend honours `seed` at all — Fireworks
        # serverless does not for chat completions (verified: same seed, temp 1.0,
        # diverges within ~40 characters; temp 0 is deterministic, so it is the
        # seed parameter being ignored, not batching noise). With arbitrary seed
        # labels the test is null by construction, so skip it rather than report
        # a p-value that means nothing.
        if v.modality == "text":
            _log(f"  seed signature skipped for {name}: "
                 f"backend does not honour `seed` for text")
            continue
        for model in sorted(set(v.model_ids)):
            mask = [i for i, m in enumerate(v.model_ids) if m == model]
            try:
                sig = seed_signature(
                    v.X[mask],
                    [v.prompt_ids[i] for i in mask],
                    [v.seeds[i] for i in mask],
                    feature_names=v.feature_names,
                    view=name,
                    model=model,
                    normalize_rows=NORMALIZE_ROWS.get(name, False),
                    n_perm=args.permutations,
                    random_state=args.random_state,
                )
            except ValueError as exc:
                _log(f"  seed signature skipped for {name}/{slug(model)}: {exc}")
                continue
            signatures.setdefault(name, []).append(sig)
            _log(
                f"  seed signature {name}/{slug(model)}: "
                f"eta2={sig.eta2_seed:.3f} F={sig.f_stat:.2f} p={sig.p_perm:.3f}"
            )

    # Experiment 2: can the prompt be recovered from the artifact?
    legibilities = {}
    for name, v in sorted(views.items()):
        leg = intent_legibility(
            v.X, v.prompt_ids, v.model_ids,
            rungs=v.rungs if v.is_ladder else None,
            view=name, modality=v.modality,
            normalize_rows=NORMALIZE_ROWS.get(name, False),
            random_state=args.random_state,
        )
        legibilities[name] = leg
        _log(f"  legibility {name}: {leg.accuracy:.3f} vs chance {leg.chance:.3f}")

    # Experiment 3: the specificity floor. Only meaningful on a ladder design.
    curves = {}
    for name, v in sorted(views.items()):
        if not v.is_ladder:
            continue
        curves[name] = rung_curve(
            v.X, v.prompt_ids, v.model_ids, v.seeds, v.rungs, v.families,
            normalize_rows=NORMALIZE_ROWS.get(name, False),
            n_boot=args.bootstrap, random_state=args.random_state,
        )
        c = curves[name]
        _log(
            f"  floor {name}: rung {c['rows'][-1]['rung']} retains "
            f"{c['floor_fraction']:.1%} of rung-1 seed variance"
        )
    if not curves:
        _log("  no ladder metadata — skipping the specificity floor curve")

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
    paths = write_report(
        root, decomps, comparison, meta,
        signatures=signatures, legibilities=legibilities, curves=curves,
    )
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


def cmd_estimate(args) -> int:
    """Count every API call the design implies, without making any of them.

    Prices are not baked in — they drift, and a stale constant that reads as
    authoritative is worse than no number. Pass --price-* to get a total.
    """
    cfg = _load_config(args)
    cfg.validate()
    counts = cfg.n_cells()
    n_img, n_txt = counts["image"], counts["text"]
    n_caption = n_img if cfg.caption_images else 0
    n_embed = n_caption + n_txt

    _log(f"design: {len(cfg.image_prompts)} image prompts x {len(cfg.image_models)} models "
         f"x {len(cfg.seeds)} seeds = {n_img} images")
    _log(f"        {len(cfg.text_prompts)} text prompts x {len(cfg.text_models)} models "
         f"x {len(cfg.seeds)} seeds = {n_txt} generations")
    _log(f"per-cell seeds: {len(cfg.seeds)} (df for the seed term: "
         f"{len(cfg.image_prompts) * len(cfg.image_models) * (len(cfg.seeds) - 1)} image, "
         f"{len(cfg.text_prompts) * len(cfg.text_models) * (len(cfg.seeds) - 1)} text)")
    print()
    print(f"  image generations   {n_img:>6}  ({cfg.image_params.steps} steps, "
          f"{cfg.image_params.width}x{cfg.image_params.height})")
    print(f"  text generations    {n_txt:>6}  (<= {cfg.text_params.max_tokens} output tokens each)")
    print(f"  VLM captions        {n_caption:>6}")
    print(f"  embedding calls     {n_embed:>6}  (batched 32/request -> "
          f"{-(-n_embed // 32)} requests)")
    print()

    if args.price_image or args.price_text_mtok or args.price_caption:
        img_cost = n_img * (args.price_image or 0.0)
        txt_cost = n_txt * cfg.text_params.max_tokens / 1e6 * (args.price_text_mtok or 0.0)
        cap_cost = n_caption * (args.price_caption or 0.0)
        print(f"  estimated cost: ${img_cost + txt_cost + cap_cost:,.2f} "
              f"(images ${img_cost:,.2f}, text ${txt_cost:,.2f}, captions ${cap_cost:,.2f})")
        print("  embeddings not priced; text cost assumes every generation hits max_tokens.")
    else:
        print("  pass --price-image / --price-text-mtok / --price-caption for a cost total")
    print("\nNothing was called. Run `generate` when you want to spend.")
    return 0


def cmd_init(args) -> int:
    cfg = ladder_config() if args.ladder else pilot_config() if args.pilot else StudyConfig()
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
        sp.add_argument("--ladder", action="store_true",
                        help="specificity-ladder design (6 rungs x 2 families)")
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

    e = sub.add_parser("estimate", help="count the API calls the design implies — calls nothing")
    common(e)
    e.add_argument("--price-image", type=float, help="$ per image generation")
    e.add_argument("--price-text-mtok", type=float, help="$ per 1M output tokens")
    e.add_argument("--price-caption", type=float, help="$ per VLM caption call")
    e.set_defaults(func=cmd_estimate)

    i = sub.add_parser("init", help="write a config file to edit")
    i.add_argument("--output", default="configs/study.json")
    i.add_argument("--pilot", action="store_true")
    i.add_argument("--ladder", action="store_true")
    i.set_defaults(func=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
