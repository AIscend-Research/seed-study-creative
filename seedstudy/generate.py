"""Run the sweep: every (prompt, model, seed) cell, once, for both modalities.

Everything except the seed is held fixed within a cell, which is what makes the
within-cell variance attributable to the seed and nothing else.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .config import StudyConfig


def slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


class Manifest:
    """Append-only record of every artifact, keyed so reruns skip finished work."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))
        self._done = {self._key(r) for r in self.rows}

    @staticmethod
    def _key(row: dict) -> tuple:
        return (row["modality"], row["model"], row["prompt_id"], row["seed"])

    def has(self, modality: str, model: str, prompt_id: str, seed: int) -> bool:
        return (modality, model, prompt_id, seed) in self._done

    def add(self, row: dict) -> None:
        with self._lock:
            if self._key(row) in self._done:
                return
            self._done.add(self._key(row))
            self.rows.append(row)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")


def _jobs(cfg: StudyConfig, modality: str) -> list[dict]:
    models = cfg.image_models if modality == "image" else cfg.text_models
    prompts = cfg.image_prompts if modality == "image" else cfg.text_prompts
    out = []
    for model in models:
        for p in prompts:
            for seed in cfg.seeds:
                out.append(
                    {
                        "modality": modality,
                        "model": model,
                        "prompt_id": p["id"],
                        "prompt": p["text"],
                        "specificity": p.get("specificity", "na"),
                        "seed": seed,
                    }
                )
    return out


def run_sweep(
    cfg: StudyConfig,
    client,
    modalities: tuple[str, ...] = ("image", "text"),
    progress=None,
) -> Manifest:
    cfg.validate()
    root = cfg.path
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(cfg.to_json())
    manifest = Manifest(root / "manifest.jsonl")

    jobs = [j for m in modalities for j in _jobs(cfg, m)]
    pending = [
        j for j in jobs if not manifest.has(j["modality"], j["model"], j["prompt_id"], j["seed"])
    ]
    skipped = len(jobs) - len(pending)
    if progress:
        progress(f"{len(jobs)} cells total, {skipped} already present, {len(pending)} to generate")

    done = 0
    errors: list[str] = []
    lock = threading.Lock()

    def work(job: dict) -> dict | None:
        nonlocal done
        art_dir = root / "artifacts" / job["modality"] / slug(job["model"]) / job["prompt_id"]
        art_dir.mkdir(parents=True, exist_ok=True)
        try:
            if job["modality"] == "image":
                data = client.generate_image(
                    model=job["model"],
                    prompt=job["prompt"],
                    seed=job["seed"],
                    **asdict(cfg.image_params),
                )
                path = art_dir / f"{job['seed']:06d}.png"
                path.write_bytes(data)
            else:
                text = client.generate_text(
                    model=job["model"],
                    prompt=job["prompt"],
                    seed=job["seed"],
                    **asdict(cfg.text_params),
                )
                path = art_dir / f"{job['seed']:06d}.txt"
                path.write_text(text)
            row = dict(job)
            row["path"] = str(path.relative_to(root))
            return row
        except Exception as exc:  # one bad cell must not sink the sweep
            with lock:
                errors.append(f"{job['modality']}/{slug(job['model'])}/{job['prompt_id']}/{job['seed']}: {exc}")
            return None
        finally:
            with lock:
                done += 1
                if progress and done % 10 == 0:
                    progress(f"  {done}/{len(pending)} generated")

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = [pool.submit(work, j) for j in pending]
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                manifest.add(row)

    if errors:
        (root / "generation_errors.txt").write_text("\n".join(errors))
        print(f"warning: {len(errors)} cells failed; see {root/'generation_errors.txt'}", file=sys.stderr)
        if progress:
            progress(f"{len(errors)} cells failed — rerun `generate` to fill the gaps")

    return manifest


def check_balance(manifest: Manifest) -> dict[str, dict]:
    """Report, per modality, whether every (prompt, model) cell has equal n.

    The decomposition needs balance. Reporting it beats silently analysing a
    lopsided grid produced by a partial sweep.
    """
    report = {}
    for modality in {r["modality"] for r in manifest.rows}:
        rows = [r for r in manifest.rows if r["modality"] == modality]
        counts: dict[tuple, int] = {}
        for r in rows:
            counts[(r["prompt_id"], r["model"])] = counts.get((r["prompt_id"], r["model"]), 0) + 1
        sizes = set(counts.values())
        report[modality] = {
            "n_artifacts": len(rows),
            "n_cells": len(counts),
            "balanced": len(sizes) == 1,
            "cell_sizes": sorted(sizes),
            "min_cell": min(counts.values()) if counts else 0,
        }
    return report
