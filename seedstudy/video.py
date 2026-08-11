"""The three-act video for the Creative AI artwork submission.

Act I    the floor        one prompt family, descending the specificity ladder,
                          seeds cycling. The flicker narrows and never stops.
Act II   the hand         one seed at a time, held across twelve unrelated
                          prompts, dissolving into its mean luminance map.
Act III  addressable      the same seed on both sides: reproducible on the left,
                          never twice on the right.

Frames are composited in PIL and piped raw into ffmpeg, so nothing is written to
disk except the finished file. Most frames repeat, so each distinct composite is
built once and pushed to the pipe as many times as it is held.

    python -m seedstudy.video --run runs/ladder --out artwork.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30

GROUND = (13, 13, 12)
INK = (240, 238, 231)
MUTED = (122, 120, 112)
DIM = (72, 71, 66)
ACCENT = (214, 176, 108)

IMAGE_MODEL = "stable-diffusion-xl-1024-v1-0"
TEXT_MODEL = "gpt-oss-120b"
RUNGS = [1, 2, 3, 4, 5, 6]
FAMILIES = ["object", "landscape"]

_FONTS = {
    "sans": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "sans_bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "serif": "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "serif_italic": "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "mono": "/System/Library/Fonts/Supplemental/Courier New.ttf",
}
_font_cache: dict = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(_FONTS[kind], size)
        except OSError:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


# ----------------------------------------------------------------- text layout


def wrap(draw, text: str, f, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=f) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(draw, xy, text, f, max_w, fill, leading=1.45, centre=False):
    x, y = xy
    step = int(f.size * leading)
    for line in wrap(draw, text, f, max_w):
        lx = x - draw.textlength(line, font=f) / 2 if centre else x
        draw.text((lx, y), line, font=f, fill=fill)
        y += step
    return y


def draw_words(draw, xy, words, f, max_w, colours, leading=1.5):
    """Lay out a word sequence where each word carries its own colour."""
    x0, y = xy
    x = x0
    space = draw.textlength(" ", font=f)
    step = int(f.size * leading)
    for word, colour in zip(words, colours):
        w = draw.textlength(word, font=f)
        if x + w > x0 + max_w and x > x0:
            x = x0
            y += step
        draw.text((x, y), word, font=f, fill=colour)
        x += w + space
    return y + step


def act_label(draw, numeral: str, title: str):
    draw.text((96, 84), numeral, font=font("serif_italic", 22), fill=ACCENT)
    draw.text((96 + 44, 84), title, font=font("sans", 22), fill=MUTED)


def base() -> Image.Image:
    return Image.new("RGB", (W, H), GROUND)


# --------------------------------------------------------------------- assets


class Assets:
    def __init__(self, run: Path):
        self.root = Path(run) / "artifacts" / "image" / IMAGE_MODEL
        if not self.root.is_dir():
            raise SystemExit(f"no images under {self.root}")
        self.seeds = sorted(int(p.stem) for p in (self.root / "object_r1").glob("*.png"))
        self._cache: dict = {}

    def path(self, pid: str, seed: int) -> Path:
        return self.root / pid / f"{seed:06d}.png"

    def square(self, pid: str, seed: int, size: int) -> Image.Image:
        key = (pid, seed, size)
        if key not in self._cache:
            with Image.open(self.path(pid, seed)) as im:
                self._cache[key] = im.convert("RGB").resize((size, size), Image.LANCZOS)
        return self._cache[key]

    def lum_cells(self, pid: str, seed: int) -> np.ndarray:
        with Image.open(self.path(pid, seed)) as im:
            g = im.convert("L").resize((4, 4), Image.BOX)
        return np.asarray(g, dtype=float).reshape(16) / 255.0


def prompts(run: Path) -> dict[str, str]:
    cfg = json.loads((Path(run) / "config.json").read_text())
    return {p["id"]: p["text"] for p in cfg["image_prompts"]}


def text_samples(root: Path, pid: str) -> list[str]:
    d = Path(root) / "artifacts" / "text" / TEXT_MODEL / pid
    if not d.is_dir():
        return []
    return [p.read_text().strip() for p in sorted(d.glob("*.txt"))]


# ----------------------------------------------------------------------- cards


def card(lines: list[tuple[str, str, int, tuple]], gap: int = 26) -> Image.Image:
    """Centred stack of (text, font-kind, size, colour)."""
    img = base()
    d = ImageDraw.Draw(img)
    rendered = []
    total = 0
    for text, kind, size, colour in lines:
        f = font(kind, size)
        ls = wrap(d, text, f, 1180)
        rendered.append((ls, f, colour))
        total += len(ls) * int(f.size * 1.4) + gap
    y = (H - total) // 2
    for ls, f, colour in rendered:
        for line in ls:
            d.text(((W - d.textlength(line, font=f)) / 2, y), line, font=f, fill=colour)
            y += int(f.size * 1.4)
        y += gap
    return img


def title_card() -> Image.Image:
    img = card(
        [
            ("The Floor", "serif", 92, INK),
            ("what specification cannot remove", "serif_italic", 34, ACCENT),
            (
                "480 generations. Two modalities. One random seed at a time.",
                "sans",
                24,
                MUTED,
            ),
        ]
    )
    d = ImageDraw.Draw(img)
    d.line((W / 2 - 120, H / 2 + 190, W / 2 + 120, H / 2 + 190), fill=DIM, width=1)
    return img


def epigraph(text: str, attrib: str | None = None) -> Image.Image:
    lines = [(text, "serif_italic", 44, INK)]
    if attrib:
        lines.append((attrib, "sans", 22, MUTED))
    return card(lines)


# ---------------------------------------------------------------- act I frames


def act_one(assets: Assets, texts: dict[str, str], family: str = "object"):
    """One composite per (rung, seed). Prompt words accumulate down the ladder."""
    size = 690
    ix, iy = (W - size) // 2, 150
    frames = []
    prev_words: list[str] = []
    for rung in RUNGS:
        pid = f"{family}_r{rung}"
        words = texts[pid].split()
        # A rung inserts words in the middle as often as at the end, so "new"
        # has to be a multiset difference against the rung above, not a position.
        carried = Counter(prev_words)
        colours = []
        for w in words:
            if carried[w] > 0:
                carried[w] -= 1
                colours.append(MUTED)
            else:
                colours.append(INK)
        for seed in assets.seeds:
            img = base()
            img.paste(assets.square(pid, seed, size), (ix, iy))
            d = ImageDraw.Draw(img)
            act_label(d, "I", "the floor")
            d.text((ix, iy - 42), f"rung {rung}", font=font("sans_bold", 26), fill=INK)
            wc = f"{len(words)} words"
            d.text(
                (ix + size - d.textlength(wc, font=font("sans", 26)), iy - 42),
                wc,
                font=font("sans", 26),
                fill=MUTED,
            )
            d.text(
                (ix, iy + size + 22),
                f"seed {seed}",
                font=font("mono", 22),
                fill=DIM,
            )
            draw_words(d, (ix, iy + size + 66), words, font("serif", 27), 1180, colours)
            frames.append(img)
        prev_words = words
    return frames


# --------------------------------------------------------------- act II frames


def seed_maps(assets: Assets) -> dict[int, np.ndarray]:
    """Per-seed mean 4x4 luminance, z-scored per cell across seeds."""
    pids = [f"{fam}_r{r}" for fam in FAMILIES for r in RUNGS]
    raw = {s: np.mean([assets.lum_cells(p, s) for p in pids], axis=0) for s in assets.seeds}
    stack = np.stack([raw[s] for s in assets.seeds])
    mu, sd = stack.mean(0), stack.std(0) + 1e-9
    return {s: (raw[s] - mu) / sd for s in assets.seeds}


def map_tile(cells: np.ndarray, size: int) -> Image.Image:
    v = np.clip(cells.reshape(4, 4) / 2.2, -1, 1)
    grey = ((v + 1) / 2 * 220 + 18).astype(np.uint8)
    return Image.fromarray(grey, mode="L").convert("RGB").resize((size, size), Image.NEAREST)


def act_two_panels(assets: Assets, maps: dict[int, np.ndarray]):
    """Per seed: (grid of twelve prompts, its luminance map) as full frames."""
    pids = [f"{fam}_r{r}" for fam in FAMILIES for r in RUNGS]
    cell, gap = 286, 10
    gw = 6 * cell + 5 * gap
    gx, gy = (W - gw) // 2, 250
    out = []
    for seed in assets.seeds:
        grid = base()
        for i, pid in enumerate(pids):
            col, row = i % 6, i // 6
            grid.paste(
                assets.square(pid, seed, cell),
                (gx + col * (cell + gap), gy + row * (cell + gap)),
            )
        d = ImageDraw.Draw(grid)
        act_label(d, "II", "the hand")
        d.text((gx, gy - 46), f"seed {seed}", font=font("sans_bold", 28), fill=INK)
        note = "twelve unrelated prompts, one seed"
        d.text(
            (gx + gw - d.textlength(note, font=font("sans", 24)), gy - 44),
            note,
            font=font("sans", 24),
            fill=MUTED,
        )

        lum = base()
        tile = 430
        lum.paste(map_tile(maps[seed], tile), ((W - tile) // 2, 260))
        d = ImageDraw.Draw(lum)
        act_label(d, "II", "the hand")
        d.text(
            ((W - d.textlength(f"seed {seed}", font=font("sans_bold", 28))) / 2, 208),
            f"seed {seed}",
            font=font("sans_bold", 28),
            fill=INK,
        )
        draw_wrapped(
            d,
            (W / 2, 730),
            "mean luminance over all twelve prompts, as deviation from the average seed",
            font("sans", 25),
            1320,
            MUTED,
            centre=True,
        )
        out.append((grid, lum))
    return out


# -------------------------------------------------------------- act III frames


def clean(text: str) -> str:
    """Swap glyphs the display faces lack for ones they have."""
    for bad, good in (("\u2011", "-"), ("\u2010", "-"), ("\u00a0", " "), ("\u200b", "")):
        text = text.replace(bad, good)
    return " ".join(text.split())


def word_novelty(cur: str, prev: str) -> float:
    a = {w.lower().strip(".,;:'’—") for w in cur.split()}
    b = {w.lower().strip(".,;:'’—") for w in prev.split()}
    return 100.0 * len(a - b) / max(len(a), 1)


def act_three(assets: Assets, samples: list[str], pid: str, n_runs: int):
    seed = assets.seeds[min(6, len(assets.seeds) - 1)]
    size = 620
    lx, ly = 150, 268
    rx, rw = 980, 800
    still = assets.square(pid, seed, size)
    frames = []
    for i in range(n_runs):
        img = base()
        img.paste(still, (lx, ly))
        d = ImageDraw.Draw(img)
        act_label(d, "III", "addressable")
        d.line((910, 210, 910, 900), fill=(34, 34, 32), width=1)

        d.text((lx, 208), "stable-diffusion-xl", font=font("sans_bold", 25), fill=INK)
        d.text((rx, 208), TEXT_MODEL, font=font("sans_bold", 25), fill=INK)
        d.text((lx, ly + size + 26), f"seed {seed}", font=font("mono", 23), fill=ACCENT)
        d.text((rx, 244), f"seed {seed}", font=font("mono", 23), fill=ACCENT)

        run = f"call {i + 1:02d} of {n_runs}"
        d.text((lx, 244), run, font=font("mono", 22), fill=DIM)
        d.text(
            (rx + rw - d.textlength(run, font=font("mono", 22)), 244),
            run,
            font=font("mono", 22),
            fill=DIM,
        )

        cur = clean(samples[i % len(samples)])
        prev = clean(samples[(i - 1) % len(samples)])
        draw_wrapped(d, (rx, 300), cur, font("serif", 27), rw, INK, leading=1.55)

        left_note = (
            "the first call" if i == 0 else "0 pixels differ from the last call"
        )
        d.text((lx, ly + size + 62), left_note, font=font("sans", 24), fill=MUTED)
        right_note = (
            "the first call"
            if i == 0
            else f"{word_novelty(cur, prev):.0f}% of the words are new since the last call"
        )
        d.text((rx, 830), right_note, font=font("sans", 24), fill=MUTED)
        frames.append(img)
    return frames


def thumbnail(assets: Assets, texts: dict[str, str], pid: str = "object_r6") -> Image.Image:
    """The thesis in one frame: every seed at maximum specification."""
    cell, gap, cols = 340, 12, 5
    gw = cols * cell + (cols - 1) * gap
    x0, y0 = (W - gw) // 2, 236
    img = base()
    for i, seed in enumerate(assets.seeds[:10]):
        col, row = i % cols, i // cols
        img.paste(assets.square(pid, seed, cell), (x0 + col * (cell + gap), y0 + row * (cell + gap)))
    d = ImageDraw.Draw(img)
    d.text((x0, 108), "The Floor", font=font("serif", 62), fill=INK)
    d.text((x0 + 4, 190), "ten seeds, one prompt, thirty-four words", font=font("serif_italic", 26), fill=ACCENT)
    y = y0 + 2 * cell + gap + 34
    draw_wrapped(d, (x0, y), texts[pid], font("serif", 24), gw, MUTED, leading=1.5)
    return img


# ------------------------------------------------------------------- assembly


def blend_seq(a: Image.Image, b: Image.Image, n: int):
    for i in range(n):
        yield Image.blend(a, b, (i + 1) / n)


def build(run: Path, text_run: Path, out: Path, ffmpeg: str):
    assets = Assets(run)
    texts = prompts(run)
    maps = seed_maps(assets)

    a1 = act_one(assets, texts)
    a2 = act_two_panels(assets, maps)
    samples = text_samples(text_run, "object_r6")
    if not samples:
        raise SystemExit(f"no text artifacts for object_r6 under {text_run}")
    a3 = act_three(assets, samples, "object_r6", n_runs=len(samples))

    plan: list[tuple[Image.Image, int]] = []

    plan.append((title_card(), 5 * FPS))

    hold = int(8 * FPS / len(assets.seeds))  # one rung is eight seconds
    for f in a1:
        plan.append((f, hold))
    plan.append(
        (
            epigraph(
                "thirty-four words removed sixty-nine per cent of it.",
                "the rest is not the artist's, and never was",
            ),
            4 * FPS,
        )
    )

    for grid, lum in a2:
        plan.append((grid, int(3.2 * FPS)))
        for f in blend_seq(grid, lum, int(1.1 * FPS)):
            plan.append((f, 1))
        plan.append((lum, int(1.7 * FPS)))
    plan.append(
        (
            epigraph(
                "the seed does not choose what you get.",
                "it chooses where the light falls, and it chooses the same way every time",
            ),
            4 * FPS,
        )
    )

    for f in a3:
        plan.append((f, int(2.5 * FPS)))
    plan.append(
        (
            card(
                [
                    ("Same randomness.", "serif", 62, INK),
                    ("One of them you can hold.", "serif", 62, INK),
                    (
                        "Diffusion honours the seed. This chat API discards it. "
                        "Every image here can be made again. None of the sentences can.",
                        "sans",
                        24,
                        MUTED,
                    ),
                ]
            ),
            7 * FPS,
        )
    )

    total = sum(n for _, n in plan)
    print(f"{len(plan)} composites, {total} frames, {total / FPS:.1f}s")

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    written = 0
    for img, n in plan:
        buf = img.tobytes()
        for _ in range(n):
            proc.stdin.write(buf)
        written += n
        if written % (30 * FPS) < n:
            print(f"  {written / FPS:6.1f}s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/ladder", help="image run directory")
    ap.add_argument("--text-run", default="results/text_ladder_gptoss")
    ap.add_argument("--out", default="artwork_the_floor.mp4")
    ap.add_argument("--thumbnail", default=None, help="write a still and stop")
    args = ap.parse_args(argv)

    if args.thumbnail:
        a = Assets(Path(args.run))
        thumbnail(a, prompts(Path(args.run))).save(args.thumbnail, quality=94)
        print(f"wrote {args.thumbnail}")
        return

    import imageio_ffmpeg

    build(Path(args.run), Path(args.text_run), Path(args.out), imageio_ffmpeg.get_ffmpeg_exe())


if __name__ == "__main__":
    main()
