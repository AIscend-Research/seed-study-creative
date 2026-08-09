"""Paper figures and a browsable gallery, built from a completed run.

Three outputs, each answering a different need:

  signature_grid   the seed-signature finding, made visible: one seed per row,
                   very different prompts across the columns. If the seed carries
                   a compositional hand, each row shares a light layout despite
                   the subjects having nothing in common.
  ladder_sheet     the specificity floor, made visible: one rung per row, seeds
                   across the columns. Rows should tighten going down — and the
                   bottom row is the randomness specification could not remove.
  gallery          every artifact, downscaled, in a self-contained page for the
                   supplement.

Compositing is done in PIL rather than matplotlib: these are image grids, not
plots, and PIL keeps the output pixel-exact with no figure-layout guesswork.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = (24, 24, 22)
MUTED = (110, 108, 102)
SURFACE = (252, 252, 251)


def _font(size: int):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:      # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def load_rows(run_dir: str | Path) -> list[dict]:
    root = Path(run_dir)
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for r in rows:
        r["_abs"] = root / r["path"]
    return [r for r in rows if r["modality"] == "image" and r["_abs"].exists()]


def _thumb(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def _lum_map(path: Path, grid: int = 4, out: int = 96) -> Image.Image:
    """The 4x4 mean-luminance grid the seed-signature test actually measures.

    Rendering the feature itself, beside the images it came from, is what turns
    'eta-squared 0.61 on lum_cell_10' into something a reader can see.
    """
    im = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float64) / 255.0
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    cell = 256 // grid
    small = lum.reshape(grid, cell, grid, cell).mean(axis=(1, 3))
    small = (small - small.min()) / (np.ptp(small) + 1e-9)
    img = Image.fromarray((small * 255).astype("uint8"), mode="L").convert("RGB")
    return img.resize((out, out), Image.NEAREST)


def _canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (w, h), SURFACE)
    return im, ImageDraw.Draw(im)


def signature_grid(
    rows: list[dict],
    out_path: str | Path,
    model: str | None = None,
    seeds: list[int] | None = None,
    prompt_ids: list[str] | None = None,
    cell: int = 180,
    show_lum: bool = True,
) -> Path:
    """Rows = seeds, columns = prompts. The signature is the row-wise consistency.

    Columns deliberately span both families: if a row still looks coherent across
    a coastline and a teapot, the consistency cannot be coming from the subject.
    """
    models = sorted({r["model"] for r in rows})
    model = model or models[0]
    sub = [r for r in rows if r["model"] == model]
    all_seeds = sorted({r["seed"] for r in sub})
    seeds = seeds or all_seeds[:5]
    if prompt_ids is None:
        prompt_ids = [
            p for p in ("landscape_r1", "landscape_r4", "landscape_r6",
                        "object_r1", "object_r4", "object_r6")
            if any(r["prompt_id"] == p for r in sub)
        ]

    index = {(r["prompt_id"], r["seed"]): r["_abs"] for r in sub}
    # `top` must clear both the subtitle at y=27 and the column labels drawn at
    # top-13, or the two run into each other.
    pad, left, top = 6, 74, 58
    lum_h = cell // 2 + 10 if show_lum else 0
    w = left + len(prompt_ids) * (cell + pad) + 12
    h = top + len(seeds) * (cell + pad) + lum_h + 46
    im, d = _canvas(w, h)
    f_lab, f_small = _font(13), _font(11)

    d.text((left, 10), f"Seed signature — {model.split('/')[-1]}", font=_font(15), fill=INK)
    d.text((left, 27), "each row is one seed held fixed across unrelated prompts",
           font=f_small, fill=MUTED)

    for c, pid in enumerate(prompt_ids):
        x = left + c * (cell + pad)
        d.text((x, top - 13), pid, font=f_small, fill=MUTED)

    for r_i, sd in enumerate(seeds):
        y = top + r_i * (cell + pad)
        d.text((6, y + cell // 2 - 7), f"seed {sd}", font=f_lab, fill=INK)
        for c, pid in enumerate(prompt_ids):
            x = left + c * (cell + pad)
            p = index.get((pid, sd))
            if p is None:
                continue
            im.paste(_thumb(p, cell), (x, y))

    if show_lum:
        y = top + len(seeds) * (cell + pad) + 8
        size = cell // 2
        d.text((6, y + size // 2 - 7), "mean", font=f_lab, fill=INK)
        d.text((6, y + size // 2 + 5), "light", font=f_lab, fill=INK)
        for r_i, sd in enumerate(seeds):
            # One averaged luminance map per seed, over every prompt in the run:
            # this is the signature isolated from subject matter.
            maps = [
                np.asarray(_lum_map(r["_abs"], out=size), dtype=np.float64)
                for r in sub
                if r["seed"] == sd
            ]
            if not maps:
                continue
            avg = np.mean(maps, axis=0)
            avg = (avg - avg.min()) / (np.ptp(avg) + 1e-9) * 255
            x = left + r_i * (size + pad)
            im.paste(Image.fromarray(avg.astype("uint8")), (x, y))
            d.text((x, y + size + 2), f"seed {sd}", font=f_small, fill=MUTED)
        d.text(
            (left, y + size + 18),
            "4x4 mean luminance per seed, averaged over all prompts — "
            "the feature the signature test scores",
            font=f_small,
            fill=MUTED,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=92)
    return out


def ladder_sheet(
    rows: list[dict],
    out_path: str | Path,
    family: str = "landscape",
    model: str | None = None,
    n_seeds: int = 8,
    cell: int = 150,
) -> Path:
    """Rows = specificity rungs, columns = seeds. The floor is the bottom row."""
    models = sorted({r["model"] for r in rows})
    model = model or models[0]
    sub = [r for r in rows if r["model"] == model and r.get("family") == family]
    rungs = sorted({r["rung"] for r in sub})
    seeds = sorted({r["seed"] for r in sub})[:n_seeds]
    index = {(r["rung"], r["seed"]): r for r in sub}

    pad, left, top = 5, 96, 44
    w = left + len(seeds) * (cell + pad) + 12
    h = top + len(rungs) * (cell + pad) + 34
    im, d = _canvas(w, h)
    f_small = _font(11)

    d.text((left, 10), f"Specificity ladder — {family}, {model.split('/')[-1]}",
           font=_font(15), fill=INK)
    d.text((left, 27), "each row adds constraints to the row above; columns are seeds",
           font=f_small, fill=MUTED)

    for r_i, rung in enumerate(rungs):
        y = top + r_i * (cell + pad)
        row = index.get((rung, seeds[0]))
        words = len(row["prompt"].split()) if row else 0
        d.text((6, y + cell // 2 - 12), f"rung {rung}", font=_font(13), fill=INK)
        d.text((6, y + cell // 2 + 2), f"{words} words", font=f_small, fill=MUTED)
        for c, sd in enumerate(seeds):
            r = index.get((rung, sd))
            if r is None:
                continue
            im.paste(_thumb(r["_abs"], cell), (left + c * (cell + pad), y))

    d.text((left, h - 22),
           "narrowing down the rows is what specification buys; whatever spread "
           "remains in the bottom row is what it cannot",
           font=f_small, fill=MUTED)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=92)
    return out


def contact_sheets(rows: list[dict], out_dir: str | Path, cell: int = 130) -> list[Path]:
    """One sheet per (model, prompt): every seed for that exact specification."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["prompt_id"]), []).append(r)

    written = []
    for (model, pid), items in sorted(groups.items()):
        items.sort(key=lambda r: r["seed"])
        cols = min(len(items), 10)
        rws = -(-len(items) // cols)
        pad, top = 4, 40
        w = cols * (cell + pad) + 12
        h = top + rws * (cell + pad) + 8
        im, d = _canvas(w, h)
        d.text((6, 8), f"{pid} — {model.split('/')[-1]}", font=_font(14), fill=INK)
        d.text((6, 25), f"{len(items)} seeds, prompt fixed", font=_font(11), fill=MUTED)
        for i, r in enumerate(items):
            x = 6 + (i % cols) * (cell + pad)
            y = top + (i // cols) * (cell + pad)
            im.paste(_thumb(r["_abs"], cell), (x, y))
        p = out_dir / f"{model.split('/')[-1]}__{pid}.jpg"
        im.save(p, quality=90)
        written.append(p)
    return written


GALLERY_CSS = """
:root{color-scheme:light dark}
body{margin:0;background:#fcfcfb;color:#0b0b0b;
 font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){body{background:#1a1a19;color:#fff}
 .card figcaption{color:#c3c2b7}.sub{color:#c3c2b7}}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 70px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:34px 0 10px}
.sub{color:#52514e;max-width:70ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.card{margin:0}
.card img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;display:block;
 background:#e3e2de}
.card figcaption{font-size:11px;color:#52514e;margin-top:4px;
 font-variant-numeric:tabular-nums}
.prompt{font-size:12px;color:#52514e;margin:0 0 8px;max-width:80ch}
"""


def gallery(
    rows: list[dict],
    out_dir: str | Path,
    thumb_px: int = 320,
    quality: int = 78,
) -> Path:
    """Self-contained browsable page over every artifact.

    Thumbnails are written beside the page rather than inlined as data URIs: 240
    inlined images would make a single file far too large to open comfortably or
    to keep in the repo.
    """
    out_dir = Path(out_dir)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)

    for r in rows:
        name = f"{r['model'].split('/')[-1]}__{r['prompt_id']}__{r['seed']:06d}.jpg"
        r["_thumb"] = f"thumbs/{name}"
        dest = out_dir / "thumbs" / name
        if not dest.exists():
            _thumb(r["_abs"], thumb_px).save(dest, quality=quality, optimize=True)

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["prompt_id"]), []).append(r)

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Seed study — image artifacts</title>",
        f"<style>{GALLERY_CSS}</style></head><body><div class='wrap'>",
        "<h1>Seed study — image artifacts</h1>",
        f"<p class='sub'>{len(rows)} generations. Within each block the prompt and "
        "model are fixed and only the seed changes, so every difference you see "
        "inside a block was produced by the seed alone.</p>",
    ]
    for (model, pid), items in sorted(groups.items()):
        items.sort(key=lambda r: r["seed"])
        parts.append(f"<h2>{html.escape(pid)} — {html.escape(model.split('/')[-1])}</h2>")
        parts.append(f"<p class='prompt'>{html.escape(items[0]['prompt'])}</p>")
        parts.append("<div class='grid'>")
        for r in items:
            parts.append(
                f"<figure class='card'><img loading='lazy' src='{r['_thumb']}' "
                f"alt='{html.escape(pid)} seed {r['seed']}'>"
                f"<figcaption>seed {r['seed']}</figcaption></figure>"
            )
        parts.append("</div>")
    parts.append("</div></body></html>")

    page = out_dir / "index.html"
    page.write_text("".join(parts))
    return page


def build_all(run_dir: str | Path, out_dir: str | Path, progress=None) -> dict[str, object]:
    rows = load_rows(run_dir)
    if not rows:
        raise ValueError(f"no image artifacts found under {run_dir}")
    out = Path(out_dir)
    made: dict[str, object] = {}

    models = sorted({r["model"] for r in rows})
    for m in models:
        tag = m.split("/")[-1]
        if progress:
            progress(f"signature grid: {tag}")
        made[f"signature_{tag}"] = signature_grid(rows, out / f"fig_signature_{tag}.jpg", model=m)
        for fam in sorted({r.get("family", "na") for r in rows}):
            if progress:
                progress(f"ladder sheet: {tag}/{fam}")
            made[f"ladder_{tag}_{fam}"] = ladder_sheet(
                rows, out / f"fig_ladder_{tag}_{fam}.jpg", family=fam, model=m
            )

    if progress:
        progress("contact sheets")
    made["contact_sheets"] = contact_sheets(rows, out / "contact_sheets")
    if progress:
        progress(f"gallery: {len(rows)} thumbnails")
    made["gallery"] = gallery(rows, out / "gallery")
    return made
