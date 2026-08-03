"""Render the decomposition as a self-contained HTML report plus a markdown summary.

Charts are hand-built SVG so the report has no runtime dependencies and can be
opened, mailed, or published as a single file.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .variance import Decomposition

# Categorical slots 1-4, in stack order, validated for adjacent-pair CVD
# separation in both modes (scripts/validate_palette.js).
SOURCES = ["seed", "prompt", "model", "prompt:model"]
LABELS = {
    "seed": "Seed",
    "prompt": "Prompt",
    "model": "Model",
    "prompt:model": "Prompt x Model",
}
LIGHT = {"seed": "#2a78d6", "prompt": "#eb6834", "model": "#1baf7a", "prompt:model": "#eda100"}
DARK = {"seed": "#3987e5", "prompt": "#d95926", "model": "#199e70", "prompt:model": "#c98500"}


def _fmt(x: float, nd: int = 3) -> str:
    if x != x:
        return "—"
    if x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f}%"


def _stacked_bars(decomps: dict[str, Decomposition]) -> str:
    """Variance share per source, one 100% stacked bar per view.

    Shares are parts of a whole that sum to 1 — the one case a stacked bar is the
    right form. A 2px surface gap separates segments; segments at or above 8%
    carry a direct label, which also discharges the light-mode contrast relief.
    """
    views = sorted(decomps, key=lambda v: (decomps[v].modality, v))
    row_h, gap, left, right, top = 46, 16, 168, 24, 12
    width = 900
    bar_w = width - left - right
    height = top + len(views) * (row_h + gap)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" width="100%" '
        f'aria-label="Share of output variance by source, per feature view">'
    ]
    for r, view in enumerate(views):
        d = decomps[view]
        y = top + r * (row_h + gap)
        parts.append(
            f'<text x="0" y="{y + 16}" class="rowlab">{html.escape(view)}</text>'
            f'<text x="0" y="{y + 32}" class="rowsub">{d.modality} · n={d.n_samples}</text>'
        )
        x = float(left)
        for src in SOURCES:
            share = d.var_share.get(src, 0.0)
            if share <= 0:
                continue
            w = share * bar_w
            draw = max(w - 2, 0.5)  # 2px surface gap between adjacent fills
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{draw:.1f}" height="{row_h - 14}" '
                f'rx="2" fill="var(--c-{src.replace(":", "-")})">'
                f"<title>{LABELS[src]}: {_pct(share)} of variance ({view})</title></rect>"
            )
            if share >= 0.08:
                parts.append(
                    f'<text x="{x + draw / 2:.1f}" y="{y + 21}" class="seg" '
                    f'text-anchor="middle">{_pct(share)}</text>'
                )
            x += w
        parts.append(
            f'<text x="{left}" y="{y + row_h - 1}" class="rowsub">'
            f"seed/prompt variance ratio {_fmt(d.seed_to_prompt, 2)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _ratio_chart(decomps: dict[str, Decomposition]) -> str:
    """Seed-to-prompt variance ratio with bootstrap interval, log scale.

    One measure, one axis. The reference line at 1.0 is the claim's hinge: to its
    right, the seed moves the output more than the prompt does.
    """
    views = sorted(decomps, key=lambda v: (decomps[v].modality, v))
    import math

    vals = []
    for v in views:
        d = decomps[v]
        lo, hi = d.seed_to_prompt_ci
        for x in (d.seed_to_prompt, lo, hi):
            if x == x and 0 < x < float("inf"):
                vals.append(x)
    if not vals:
        return "<p class='muted'>No finite ratios to plot.</p>"
    lo_b, hi_b = min(vals + [0.5]), max(vals + [2.0])
    lo_l, hi_l = math.log10(lo_b * 0.7), math.log10(hi_b * 1.4)

    left, right, top, row_h = 168, 40, 28, 34
    width = 900
    plot_w = width - left - right
    height = top + len(views) * row_h + 34

    def px(x: float) -> float:
        x = min(max(x, 10**lo_l), 10**hi_l)
        return left + (math.log10(x) - lo_l) / (hi_l - lo_l) * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" width="100%" '
        f'aria-label="Seed-to-prompt variance ratio with 95% bootstrap interval">'
    ]
    ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 4, 10, 25, 100) if 10**lo_l <= t <= 10**hi_l]
    for t in ticks:
        parts.append(
            f'<line x1="{px(t):.1f}" y1="{top - 8}" x2="{px(t):.1f}" y2="{height - 30}" '
            f'class="{"ref" if t == 1 else "grid"}"/>'
            f'<text x="{px(t):.1f}" y="{height - 14}" class="tick" text-anchor="middle">{t:g}</text>'
        )
    for r, view in enumerate(views):
        d = decomps[view]
        y = top + r * row_h + row_h / 2
        parts.append(f'<text x="0" y="{y + 4:.0f}" class="rowlab">{html.escape(view)}</text>')
        lo, hi = d.seed_to_prompt_ci
        if lo == lo and hi == hi:
            parts.append(
                f'<line x1="{px(lo):.1f}" y1="{y:.1f}" x2="{px(hi):.1f}" y2="{y:.1f}" class="ci"/>'
            )
        val = d.seed_to_prompt
        if val == val:
            capped = val == float("inf")
            parts.append(
                f'<circle cx="{px(val if not capped else 10**hi_l):.1f}" cy="{y:.1f}" r="5.5" '
                f'class="dot"><title>{html.escape(view)}: ratio '
                f'{_fmt(val, 2)} (95% CI {_fmt(lo, 2)}–{_fmt(hi, 2)})</title></circle>'
            )
    parts.append(
        f'<text x="{left}" y="{top - 14}" class="rowsub">'
        "seed variance / prompt variance — right of 1.0, the seed dominates</text></svg>"
    )
    return "".join(parts)


def _table(decomps: dict[str, Decomposition]) -> str:
    rows = []
    for view in sorted(decomps, key=lambda v: (decomps[v].modality, v)):
        d = decomps[view]
        for row in d.table:
            rows.append(
                "<tr>"
                f"<td>{html.escape(view)}</td><td>{d.modality}</td>"
                f"<td>{LABELS[row['source']]}</td><td class='num'>{row['df']}</td>"
                f"<td class='num'>{row['ss']:.1f}</td>"
                f"<td class='num'>{row['ms']:.2f}</td>"
                f"<td class='num'>{_pct(row['eta2'])}</td>"
                f"<td class='num'>{row['var_component']:.4f}</td>"
                f"<td class='num'>{_pct(row['var_share'])}</td>"
                f"<td class='num'>{_fmt(row['p_perm'])}</td>"
                "</tr>"
            )
    return (
        "<table><thead><tr><th>view</th><th>modality</th><th>source</th><th>df</th>"
        "<th>SS</th><th>MS</th><th>&eta;&sup2;</th><th>var. component</th>"
        "<th>var. share</th><th>p (perm)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _floor_chart(curves: dict) -> str:
    """Seed variance against specificity rung — change over an ordered x, so a line.

    The story is the shape, not the levels: if the curves flatten above zero,
    specification has a limit and the flat part is what prompting cannot buy.
    """
    if not curves:
        return ""
    views = sorted(curves)
    left, right, top, bottom = 56, 150, 20, 40
    width, height = 900, 330
    plot_w, plot_h = width - left - right, height - top - bottom

    all_rows = [r for v in views for r in curves[v]["rows"]]
    rungs = sorted({r["rung"] for r in all_rows})
    ymax = max([r["ci_hi"] if r["ci_hi"] == r["ci_hi"] else r["seed_variance"] for r in all_rows] + [0])
    ymax = ymax * 1.12 or 1.0

    def px(rung):
        i = rungs.index(rung)
        return left + (i / max(len(rungs) - 1, 1)) * plot_w

    def py(v):
        return top + plot_h - (v / ymax) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" width="100%" '
        f'aria-label="Seed variance by prompt specificity rung">'
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{frac * ymax:.2f}</text>"
        )
    for r in rungs:
        parts.append(
            f'<text x="{px(r):.1f}" y="{height - 18}" class="tick" text-anchor="middle">'
            f"rung {r}</text>"
        )
    # End-of-line labels collide when two views land at similar variance, so
    # place them first and push overlapping ones apart before drawing.
    ends = sorted(
        (
            (sorted(curves[v]["rows"], key=lambda r: r["rung"])[-1]["seed_variance"], v)
            for v in views
        ),
        key=lambda t: -t[0],
    )
    label_y: dict[str, float] = {}
    prev = -1e9
    for value, view in ends:
        y = max(py(value), prev + 14)
        label_y[view] = y
        prev = y

    for k, view in enumerate(views):
        slot = SOURCES[k % len(SOURCES)]
        colour = f'var(--c-{slot.replace(":", "-")})'
        rows = sorted(curves[view]["rows"], key=lambda r: r["rung"])
        band = " ".join(f"{px(r['rung']):.1f},{py(r['ci_hi']):.1f}" for r in rows if r["ci_hi"] == r["ci_hi"])
        band2 = " ".join(
            f"{px(r['rung']):.1f},{py(r['ci_lo']):.1f}" for r in reversed(rows) if r["ci_lo"] == r["ci_lo"]
        )
        if band and band2:
            parts.append(f'<polygon points="{band} {band2}" fill="{colour}" opacity="0.13"/>')
        pts = " ".join(f"{px(r['rung']):.1f},{py(r['seed_variance']):.1f}" for r in rows)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for r in rows:
            parts.append(
                f'<circle cx="{px(r["rung"]):.1f}" cy="{py(r["seed_variance"]):.1f}" r="4" '
                f'fill="{colour}" stroke="var(--surface-1)" stroke-width="2">'
                f'<title>{html.escape(view)} rung {r["rung"]}: seed variance '
                f'{r["seed_variance"]:.3f} ({_pct(r["fraction_of_rung1"])} of rung 1)</title></circle>'
            )
        last = rows[-1]
        ly = label_y[view]
        parts.append(
            f'<line x1="{px(last["rung"]):.1f}" y1="{py(last["seed_variance"]):.1f}" '
            f'x2="{px(last["rung"]) + 8:.1f}" y2="{ly:.1f}" stroke="{colour}" '
            f'stroke-width="1" opacity="0.5"/>'
            f'<text x="{px(last["rung"]) + 12:.1f}" y="{ly + 4:.1f}" '
            f'class="rowsub">{html.escape(view)}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{top - 6}" class="rowsub">seed variance (standardised units)</text></svg>'
    )
    return "".join(parts)


def _legibility_chart(legs: dict) -> str:
    """Recovery accuracy per view against its chance line — one measure, one axis."""
    if not legs:
        return ""
    views = sorted(legs, key=lambda v: (legs[v].modality, v))
    left, right, top, row_h = 168, 60, 16, 38
    width = 900
    plot_w = width - left - right
    height = top + len(views) * row_h + 34

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" width="100%" '
        f'aria-label="Prompt-recovery accuracy per feature view against chance">'
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + frac * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 30}" class="grid"/>'
            f'<text x="{x:.1f}" y="{height - 14}" class="tick" text-anchor="middle">'
            f"{int(frac * 100)}%</text>"
        )
    for r, view in enumerate(views):
        leg = legs[view]
        y = top + r * row_h
        parts.append(f'<text x="0" y="{y + 20}" class="rowlab">{html.escape(view)}</text>')
        w = max(leg.accuracy * plot_w - 2, 0.5)
        cx = left + leg.chance * plot_w
        # Clear the chance line so a short bar's value label doesn't sit on it.
        label_x = max(left + w + 8, cx + 10)
        parts.append(
            f'<rect x="{left}" y="{y + 6}" width="{w:.1f}" height="18" rx="2" '
            f'fill="var(--c-seed)"><title>{html.escape(view)}: recovered '
            f"{_pct(leg.accuracy)} of the time, chance {_pct(leg.chance)}</title></rect>"
            f'<line x1="{cx:.1f}" y1="{y + 2}" x2="{cx:.1f}" y2="{y + 28}" class="ref"/>'
            f'<text x="{label_x:.1f}" y="{y + 20}" class="rowsub">{_pct(leg.accuracy)}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{height - 2}" class="rowsub">'
        "dashed line marks chance for that view&#8217;s prompt count</text></svg>"
    )
    return "".join(parts)


def _signature_table(signatures: dict) -> str:
    rows = []
    for view in sorted(signatures):
        for sig in signatures[view]:
            feats = ", ".join(f["feature"] for f in sig.top_features[:3]) or "—"
            rows.append(
                "<tr>"
                f"<td>{html.escape(view)}</td>"
                f"<td>{html.escape(sig.model.split('/')[-1])}</td>"
                f"<td class='num'>{_pct(sig.eta2_seed)}</td>"
                f"<td class='num'>{_pct(sig.eta2_prompt)}</td>"
                f"<td class='num'>{_fmt(sig.f_stat, 2)}</td>"
                f"<td class='num'>{_fmt(sig.p_perm)}</td>"
                f"<td>{html.escape(feats)}</td>"
                "</tr>"
            )
    if not rows:
        return "<p class='muted'>No seed-signature tests were run.</p>"
    return (
        "<table><thead><tr><th>view</th><th>model</th><th>&eta;&sup2; seed</th>"
        "<th>&eta;&sup2; prompt</th><th>F</th><th>p (perm)</th>"
        "<th>features the seed acts on</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


CSS = """
:root { color-scheme: light dark; }
.viz-root {
  --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e; --line:#e3e2de;
  --c-seed:#2a78d6; --c-prompt:#eb6834; --c-model:#1baf7a; --c-prompt-model:#eda100;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7; --line:#3a3a37;
    --c-seed:#3987e5; --c-prompt:#d95926; --c-model:#199e70; --c-prompt-model:#c98500;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7; --line:#3a3a37;
  --c-seed:#3987e5; --c-prompt:#d95926; --c-model:#199e70; --c-prompt-model:#c98500;
}
body { margin:0; background:var(--surface-1); color:var(--text-primary);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1000px; margin:0 auto; padding:40px 24px 80px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:40px 0 10px; }
p, li { color:var(--text-secondary); max-width:74ch; }
.lede { color:var(--text-primary); }
.headline { font-size:44px; font-weight:650; letter-spacing:-0.02em; margin:8px 0 0; }
.headline small { display:block; font-size:14px; font-weight:400; color:var(--text-secondary); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:24px 28px; margin:20px 0 8px; }
.tile { min-width:0; }
figure { margin:0 0 8px; }
figcaption { font-size:13px; color:var(--text-secondary); margin-top:6px; }
.rowlab { fill:var(--text-primary); font-size:13px; font-weight:550; }
.rowsub, .tick { fill:var(--text-secondary); font-size:11px; }
.seg { fill:#fff; font-size:11px; font-weight:600; }
.grid { stroke:var(--line); stroke-width:1; }
.ref  { stroke:var(--text-secondary); stroke-width:1.5; stroke-dasharray:3 3; }
.ci   { stroke:var(--c-seed); stroke-width:2; stroke-linecap:round; opacity:.45; }
.dot  { fill:var(--c-seed); stroke:var(--surface-1); stroke-width:2; }
.legend { display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 0; font-size:13px;
  color:var(--text-secondary); }
.swatch { width:11px; height:11px; border-radius:2px; display:inline-block;
  margin-right:6px; vertical-align:-1px; }
.tblwrap { overflow-x:auto; }
table { border-collapse:collapse; font-size:13px; min-width:760px; }
th, td { text-align:left; padding:6px 12px 6px 0; border-bottom:1px solid var(--line);
  white-space:nowrap; }
th { color:var(--text-secondary); font-weight:550; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.muted { color:var(--text-secondary); font-size:13px; }
"""


def _experiment_sections(signatures: dict, legibilities: dict, curves: dict) -> str:
    out = []

    if curves:
        floors = [
            f"{v}: {_pct(c['floor_fraction'])}" for v, c in sorted(curves.items())
        ]
        out.append(
            "<h2>Does specification have a floor?</h2>"
            "<p>Each rung adds constraints to the rung below it and changes nothing else, "
            "so the curve isolates what specification buys. Seed variance retained at the "
            f"top rung, relative to a three-word prompt: {'; '.join(floors)}. Whatever the "
            "curve flattens to is the randomness prompting cannot remove.</p>"
            f"<figure>{_floor_chart(curves)}"
            "<figcaption>Line is mean within-cell seed variance in standardised units; band "
            "is the 95% bootstrap interval over seeds. Features are standardised once across "
            "all rungs, so the rungs share a space and the levels are comparable. No prompt "
            "variance appears in this figure — the claim is absolute, so it does not depend "
            "on which prompts were chosen.</figcaption></figure>"
        )

    if legibilities:
        best = max(legibilities.values(), key=lambda l: l.accuracy)
        out.append(
            "<h2>Can the prompt be recovered from the artifact?</h2>"
            "<p>A leave-one-out nearest-centroid classifier predicts which prompt produced "
            "each output, run within a generative model so model identity cannot leak in as "
            f"a cue. Best view recovers the prompt {_pct(best.accuracy)} of the time against "
            f"chance of {_pct(best.chance)}. Specification that cannot be read back off the "
            "work is specification that did not survive into it.</p>"
            f"<figure>{_legibility_chart(legibilities)}</figure>"
        )
        rungs = [l for l in legibilities.values() if l.per_rung]
        if rungs:
            cells = []
            for l in sorted(rungs, key=lambda x: x.view):
                acc = ", ".join(f"r{r['rung']} {_pct(r['accuracy'])}" for r in l.per_rung)
                cells.append(f"<li><strong>{html.escape(l.view)}</strong> — {acc}</li>")
            out.append(
                "<p>Legibility by rung — whether adding words makes intent more readable:</p>"
                f"<ul>{''.join(cells)}</ul>"
            )

    out.append(
        "<h2>Is the seed noise, or a style?</h2>"
        "<p>The decomposition treats the seed as a replicate: seed 7 for one prompt is "
        "assumed to share nothing with seed 7 for another. That assumption is testable, "
        "because the same seeds run against every prompt. Prompt and seed form a complete "
        "crossed grid with one artifact per cell, so the prompt&times;seed interaction is "
        "the error term and F = MS<sub>seed</sub> / MS<sub>residual</sub>. A seed effect "
        "that survives is not noise — it is a consistent hand, applied on top of whatever "
        "was asked for.</p>"
        f"<div class='tblwrap'>{_signature_table(signatures)}</div>"
        "<p class='muted'>Seed labels are permuted independently within each prompt, which "
        "breaks cross-prompt consistency while leaving each prompt&#8217;s own spread intact. "
        "A null result here supports the main decomposition rather than undermining it.</p>"
    )
    return "".join(out)


def render_html(
    decomps: dict[str, Decomposition],
    comparison: dict,
    meta: dict,
    signatures: dict | None = None,
    legibilities: dict | None = None,
    curves: dict | None = None,
) -> str:
    legend = "".join(
        f'<span><i class="swatch" style="background:var(--c-{s.replace(":", "-")})"></i>'
        f"{LABELS[s]}</span>"
        for s in SOURCES
    )
    sem = comparison.get("semantic_pair")
    surf = comparison.get("surface_pair")

    tiles = []
    for view in ("image_semantic", "text_semantic", "image_lowlevel", "text_lexical"):
        if view in decomps:
            d = decomps[view]
            tiles.append(
                f'<div class="tile"><div class="headline">{_pct(d.var_share["seed"])}'
                f"<small>{view} — variance from seed alone<br>prompt: "
                f'{_pct(d.var_share["prompt"])}</small></div></div>'
            )

    if sem:
        verdict = (
            f"In the shared embedding space the seed accounts for "
            f"<strong>{_pct(sem['image_seed_share'])}</strong> of image variance and "
            f"<strong>{_pct(sem['text_seed_share'])}</strong> of text variance."
        )
    else:
        verdict = "Semantic views were not built for this run."

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seed study — how much of the output is the seed?</title>
<style>{CSS}</style></head>
<body class="viz-root"><div class="wrap">
<h1>How much of a generated artifact is the seed?</h1>
<p class="lede">Prompt and model held fixed; seed swept. Everything that moves inside a
(prompt, model) cell moved because the seed moved. {verdict}</p>

<div class="tiles">{"".join(tiles)}</div>

<h2>Share of output variance by source</h2>
<figure>{_stacked_bars(decomps)}
<div class="legend">{legend}</div>
<figcaption>EMS-corrected variance components, normalised to sum to 1. Shares are
scale-free, which is what makes the image and text rows comparable.</figcaption></figure>

<h2>Seed-to-prompt variance ratio</h2>
<figure>{_ratio_chart(decomps)}
<figcaption>Dot is the point estimate; line is the 95% two-level bootstrap interval
(resampling prompts and seeds). Intervals that straddle 1.0 do not establish
seed dominance for that view.</figcaption></figure>

{_experiment_sections(signatures or {}, legibilities or {}, curves or {})}

<h2>Full decomposition</h2>
<div class="tblwrap">{_table(decomps)}</div>
<p class="muted">&eta;&sup2; is the share of variance in this sample and is inflated for
prompt and model, whose mean squares carry seed noise. The variance component column
removes that inflation and is the number the headline rests on. Seed has no
permutation test: it is the residual stratum, with no labels to shuffle.</p>

<h2>Cross-modal comparison</h2>
<p>{
  "Semantic pair (same embedding space): image seed share "
  + _pct(sem["image_seed_share"]) + " vs text seed share " + _pct(sem["text_seed_share"])
  + f", a gap of {_pct(abs(sem['seed_share_gap']))}." if sem else "Not available."
}</p>
<p>{
  "Surface pair (model-free controls): image " + _pct(surf["image_seed_share"])
  + " vs text " + _pct(surf["text_seed_share"])
  + ". Agreement with the semantic pair means the result is not an artifact of the embedder."
  if surf else ""
}</p>

<h2>Run</h2>
<div class="tblwrap"><table><tbody>{"".join(
    f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
    for k, v in meta.items())}</tbody></table></div>
</div></body></html>"""


def render_markdown(
    decomps: dict[str, Decomposition],
    comparison: dict,
    meta: dict,
    signatures: dict | None = None,
    legibilities: dict | None = None,
    curves: dict | None = None,
) -> str:
    lines = ["# Seed study — results", ""]
    lines.append(f"Run: `{meta.get('run', '?')}` · {meta.get('n_artifacts', '?')} artifacts")
    lines.append("")
    lines.append("| view | modality | seed | prompt | model | prompt x model | seed/prompt | 95% CI | P(seed pair > prompt pair) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for view in sorted(decomps, key=lambda v: (decomps[v].modality, v)):
        d = decomps[view]
        lo, hi = d.seed_to_prompt_ci
        lines.append(
            f"| {view} | {d.modality} | {_pct(d.var_share['seed'])} | "
            f"{_pct(d.var_share['prompt'])} | {_pct(d.var_share['model'])} | "
            f"{_pct(d.var_share['prompt:model'])} | {_fmt(d.seed_to_prompt, 2)} | "
            f"{_fmt(lo, 2)}–{_fmt(hi, 2)} | {_fmt(d.p_seed_beats_prompt, 2)} |"
        )
    if curves:
        lines += ["", "## Specificity floor (seed variance by rung)", ""]
        lines.append("| view | " + " | ".join(
            f"r{r['rung']}" for r in sorted(next(iter(curves.values()))["rows"], key=lambda x: x["rung"])
        ) + " | retained at top |")
        lines.append("|---" * (len(next(iter(curves.values()))["rows"]) + 2) + "|")
        for view, c in sorted(curves.items()):
            vals = " | ".join(
                _pct(r["fraction_of_rung1"]) for r in sorted(c["rows"], key=lambda x: x["rung"])
            )
            lines.append(f"| {view} | {vals} | **{_pct(c['floor_fraction'])}** |")
        lines.append("")
        lines.append("Values are seed variance as a fraction of the three-word prompt's.")

    if legibilities:
        lines += ["", "## Intent legibility (prompt recovered from artifact)", ""]
        lines.append("| view | accuracy | chance | lift | 95% CI |")
        lines.append("|---|---|---|---|---|")
        for view, l in sorted(legibilities.items()):
            lines.append(
                f"| {view} | {_pct(l.accuracy)} | {_pct(l.chance)} | {_pct(l.lift)} | "
                f"{_pct(l.ci_lo)}–{_pct(l.ci_hi)} |"
            )

    if signatures:
        lines += ["", "## Seed signature (is the seed a style?)", ""]
        lines.append("| view | model | eta2 seed | eta2 prompt | F | p | acts on |")
        lines.append("|---|---|---|---|---|---|---|")
        for view in sorted(signatures):
            for sig in signatures[view]:
                feats = ", ".join(f["feature"] for f in sig.top_features[:3]) or "—"
                lines.append(
                    f"| {view} | {sig.model.split('/')[-1]} | {_pct(sig.eta2_seed)} | "
                    f"{_pct(sig.eta2_prompt)} | {_fmt(sig.f_stat, 2)} | {_fmt(sig.p_perm)} | {feats} |"
                )

    sem = comparison.get("semantic_pair")
    if sem:
        lines += [
            "",
            "## Cross-modal (shared embedding space)",
            "",
            f"- image seed share: {_pct(sem['image_seed_share'])}",
            f"- text seed share: {_pct(sem['text_seed_share'])}",
            f"- gap: {_pct(abs(sem['seed_share_gap']))}",
        ]
    lines += [
        "",
        "## Reading this",
        "",
        "- Seed variance is within-cell variance: prompt and model are fixed inside a cell,",
        "  so the seed is the only thing left that could have moved the output.",
        "- Variance shares are EMS-corrected components, not raw eta-squared; raw eta-squared",
        "  credits the prompt with seed noise carried in its mean square.",
        "- Prompts are resampled in the bootstrap, so the interval reflects the dominant",
        "  uncertainty: which prompts were chosen, not how many seeds were run.",
    ]
    return "\n".join(lines) + "\n"


def write_report(
    out_dir: str | Path,
    decomps: dict[str, Decomposition],
    comparison: dict,
    meta: dict,
    signatures: dict | None = None,
    legibilities: dict | None = None,
    curves: dict | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": out / "report.html",
        "markdown": out / "report.md",
        "json": out / "results.json",
    }
    paths["html"].write_text(
        render_html(decomps, comparison, meta, signatures, legibilities, curves)
    )
    paths["markdown"].write_text(
        render_markdown(decomps, comparison, meta, signatures, legibilities, curves)
    )
    paths["json"].write_text(
        json.dumps(
            {
                "meta": meta,
                "decompositions": {k: v.to_dict() for k, v in decomps.items()},
                "comparison": comparison,
                "seed_signatures": {
                    k: [s.to_dict() for s in v] for k, v in (signatures or {}).items()
                },
                "legibility": {k: v.to_dict() for k, v in (legibilities or {}).items()},
                "specificity_floor": curves or {},
            },
            indent=2,
        )
    )
    return paths
