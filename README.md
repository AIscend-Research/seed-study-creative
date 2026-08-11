# seed-study

How much of a generated artifact is determined by the random seed alone, before
any human intent expressed in the prompt has a chance to matter? And does the
answer look the same for images and for text?

Fix the prompt. Fix the model. Sweep the seed. Decompose the variance. Do it in
both modalities.

Everything runs on [Fireworks](https://fireworks.ai): image generation, text
generation, VLM captioning, and embeddings all come from the same provider, so
the two modalities are measured through comparable machinery.

**Status (2026-08-10): both sweeps have run. 240 image artifacts + 240 text
artifacts, analysed, committed under `results/`.** Everything below the
"Paper kit" section is method documentation; the paper kit is the writing brief.

---

# Paper kit

Everything needed to write the NeurIPS 2026 Creative AI track paper (non-archival,
2–6 pages excluding references, single-blind, deadline 10 August AoE). Numbers are
copied from `results/*/results.json`; figures are in `results/figures/` and
bundled as `seedstudy_figures_images.zip`.

## 1. The claim to argue

The study set out to show that seed variance rivals or exceeds prompt variance,
that the prompt is credited with authorial work the seed is actually doing. **That
hypothesis lost.** In the semantic view the prompt dominates decisively in both
modalities, and the prompt is recoverable from the artifact at 8–12× chance. Do
not write the paper it was designed to write.

The result that survived is better suited to the track's *Agency* theme, and it is
three-part:

1. **Specification has a floor.** Going from three words to thirty-five removes
   ~69% of semantic randomness in images and then stops. The remaining ~31% is
   what no amount of further specification buys back. In text the semantic floor
   never descends at all.
2. **The residue is not noise; it is a hand.** The image seed has a
   *cross-prompt consistent* effect (p ≤ 0.002, both models), and the features it
   acts on are spatial luminance cells. The seed does not choose what you get; it
   chooses how it is lit and arranged, and it makes the same choice every time you
   reuse it. That is the profile of a collaborator with a style, not a
   perturbation.
3. **That collaborator is addressable in one modality and not the other.**
   Diffusion honours the seed: the same integer reproduces the same image, so the
   randomness can be held, returned to, and used. Fireworks' serverless chat API
   ignores `seed`: the text randomness is real, comparably sized, and
   *unaddressable*: no text generation here can be reproduced. Where the residue
   is indexable, an artist can take it up as material; where it is not, the same
   residue is only attrition.

Framed for the theme: agency does not end where the prompt ends. There is a
measurable band of authorship below specification, and whether it is available to
the artist is decided by an infrastructural detail (whether the API passes the
seed through), not by the artist, the model, or the prompt.

## 2. Headline numbers

### Variance decomposition (EMS-corrected variance components, as shares)

| view | prompt | model | prompt×model | seed | seed/prompt (95% CI) | P(seed pair > prompt pair) |
|---|---|---|---|---|---|---|
| `image_semantic` | **62.4%** | 1.3% | 5.4% | 30.9% | 0.50 (0.31–0.67) | 0.117 |
| `image_lowlevel` | 25.9% | 26.3% | 12.3% | **35.5%** | 1.37 (0.84–2.19) | 0.281 |
| `text_semantic` | **76.3%** | 0.0%* | 0.0%* | 23.7% | 0.31 (0.23–0.38) | 0.016 |
| `text_lexical` | **78.2%** | 0.0%* | 0.0%* | 21.8% | 0.28 (0.17–0.36) | 0.136 |

\* single-model fallback, not a finding: only one text model survived (see §5).

The semantic CIs exclude 1.0 in both modalities: the prompt wins, and not
narrowly. The low-level image CI straddles 1.0, so seed dominance is not
established there either. Report this honestly and early; it is the load-bearing
negative result.

### Specificity floor: seed variance per rung, as a fraction of the three-word prompt's

| view | r1 (2–3 w) | r2 | r3 | r4 | r5 | r6 (34 w) |
|---|---|---|---|---|---|---|
| `image_semantic` | 100% | 70.0% | 53.8% | 44.4% | 50.4% | **30.9%** |
| `image_lowlevel` | 100% | 114.7% | 116.7% | 68.0% | 76.3% | **55.9%** |
| `text_semantic` | 100% | 102.4% | 103.3% | 120.5% | 108.8% | **86.6%** |
| `text_lexical` | 100% | 96.1% | 32.6% | 49.7% | 55.4% | **50.7%** |

Rung-6 bootstrap band, `image_semantic`: 0.123–0.199 (absolute per-dimension
variance; the rung-1 value is 0.566). It sits well clear of zero: the floor is
not a measurement limit.

**The text semantic row is the paper's most arguable result.** Six rungs of added
constraint, ending at 35 words of pinned-down detail with sentence count and
tense specified, leave semantic within-cell variance roughly where three words
left it, while the *lexical* floor halves over the same range. The constraints
do regularise length and sentence shape; the semantic spread does not move. One
model, so present it as a hypothesis, not a result.

### Intent legibility: leave-one-out nearest-centroid recovery of which of 12 prompts produced an artifact (chance 8.3%)

| view | accuracy | 95% CI | per-rung (r1→r6) |
|---|---|---|---|
| `image_semantic` | 74.6% | 68.8–80.4% | 72.5, 82.5, 87.5, 77.5, 42.5, 85.0 |
| `image_lowlevel` | 67.9% | 62.1–73.8% | 67.5, 57.5, 67.5, 75.0, 60.0, 80.0 |
| `text_semantic` | **99.6%** | n/a | 100, 97.5, 100, 100, 100, 100 |
| `text_lexical` | 72.5% | 66.7–78.3% | 67.5, 70.0, 72.5, 60.0, 80.0, 85.0 |

Per-model image accuracy: playground-v2 79.2% / SDXL 70.0% (semantic).
This is the sentence that kills the original hypothesis: whatever the seed is
doing, it is not erasing the specification.

### Seed signature: is the seed's effect consistent *across unrelated prompts*?

Randomised-block ANOVA with seed labels permuted independently within each
prompt, so only cross-prompt consistency is tested. Run per model.

| view | model | η²_seed | η²_prompt | F | p (999 perms) |
|---|---|---|---|---|---|
| `image_lowlevel` | stable-diffusion-xl | **16.8%** | 47.8% | 5.24 | 0.001 |
| `image_lowlevel` | playground-v2 | **10.1%** | 62.6% | 4.08 | 0.001 |
| `image_semantic` | stable-diffusion-xl | 4.0% | 68.0% | n/a | 0.001 |
| `image_semantic` | playground-v2 | 2.9% | 71.2% | n/a | 0.002 |
| text (either view) | n/a | *not run* | | | skipped: seed labels are arbitrary (§5) |

Top features, `image_lowlevel`. **Every one is a spatial luminance cell**:

- SDXL: `lum_cell_10` (η²=.61, strongest seed 7), `lum_cell_7` (.49, seed 3), `lum_cell_0` (.49, seed 8), `lum_cell_3` (.44), `lum_cell_1` (.43), `lum_cell_4` (.43)
- playground-v2: `lum_cell_4` (.48, seed 4), `lum_cell_0` (.36, seed 8), `lum_cell_8` (.31, seed 8), `lum_cell_10` (.30, seed 7)

Semantic top features are anonymous embedding dimensions (`e261`, `e109`, …) and
carry no interpretation; cite the low-level ones only.

Note the asymmetry: strong low-level (η² 10–17%), weak semantic (η² 3–4%,
significant but small). The seed's consistent hand is **compositional, not
semantic**. Mechanistically unsurprising (the initial latent most directly
constrains global structure), but "expected" is not "uninteresting": the finding
is that the constraint holds *across every prompt*, which is what makes it a
style rather than a perturbation.

## 3. The figures, and how to describe them

Zipped for portability: **`seedstudy_figures_images.zip`** (6.8 MB, 275 files:
the paper-ready figures, all contact sheets, and the browsable gallery). The raw
240 PNGs are the separate 295 MB `seedstudy_ladder-image_240img_2026-08-06.zip`
and are not needed to write.

### `fig_ladder_<model>_<family>.jpg`, 4 files. **Lead figure.**

A 6×8 grid. Rows are specificity rungs, labelled at left with the word count
(`rung 1 / 2 words` → `rung 6 / 34 words`); each row's prompt strictly extends the
row above. Columns are eight fixed seeds, so a column is one seed held constant
down the whole ladder. The title bar names the family and model; the
footer reads *"narrowing down the rows is what specification buys; whatever
spread remains in the bottom row is what it cannot."*

What a reader sees, in the `object`/SDXL variant: row 1 (`"a teapot"`) is
chaotic: an oxblood-glazed pot, a Japanese-scene enamel, a green transferware,
a pewter one, a blue-and-white Delft, a dark grey one, several with stray cups
and fruit in frame. By row 3 (`"a blue ceramic teapot on a table"`) every pot is
the same mid-blue; the remaining variation is staging: a plate, a plant, a pair
of cups, a wooden tray. By row 6 the backdrops are uniformly white and seamless,
the lighting is the same soft frontal wash, and what is *still* varying is pot
silhouette, scale within the frame, the angle of the handle and spout, and where
the specular highlight lands. **That bottom row is the visual argument of the
paper: the floor is a picture, not a number.** The `landscape` family is the same
story with palette collapsing toward muted grey-blue and the lighthouse arriving
at rung 4.

Suggested caption: *"Specificity ladder (SDXL, object family). Rows add
constraints; columns are fixed seeds. Rows 1–3 show what specification removes.
Row 6, at 34 words pinned to a seamless backdrop and a lighting setup, is what it
cannot: silhouette, scale, viewpoint, and where the light lands remain the seed's
to decide. Semantic within-cell variance at row 6 is 31% of row 1's."*

### `fig_signature_<model>.jpg`, 2 files. **The finding figure.**

A 5×6 grid plus a strip underneath. Each **row is one seed held fixed across six
unrelated prompts** (columns: `landscape_r1`, `landscape_r4`, `landscape_r6`,
`object_r1`, `object_r4`, `object_r6`). Beneath the grid sits a row of five 4×4
greyscale blocks labelled *"mean light"*: the per-seed mean luminance map,
averaged over all twelve prompts.

Be honest about what the images themselves show: **mostly the prompt winning.**
Every cell in the `landscape_r6` column is a grey-blue coastline; every cell in
`object_r6` is a blue teapot on white. Reading a per-seed personality off the
photographs is hard, which is exactly what η²_semantic = 3–4% predicts. **The
claim is carried by the luminance strip**, where the differences are plain: seed
1's map is bright top-left with a dark lower-middle; seed 2 is bright-right and
dark-centre; seed 5 is nearly black in the lower-left quadrant. Those are the
`lum_cell_*` features the signature test scores, and they persist across a
lighthouse and a teapot alike.

Suggested caption: *"Seed signature (SDXL). Each row is one seed across six
unrelated prompts; the strip below is that seed's mean 4×4 luminance map,
averaged over all prompts. The photographs show the prompt dominating (η²_seed =
4% semantically); the strip shows what survives it (η²_seed = 17% on low-level
features, p = 0.001, seed labels permuted within prompt). The seed's consistent
contribution is where the light falls."*

### `figures/contact_sheets/`, 24 files

One sheet per (model, prompt): every seed at that exact specification, named
`<model>__<family>_r<rung>.jpg`. Use for an appendix, or to pull a single cell if
a reviewer disputes a rung. Not needed in the main body.

### `figures/gallery/index.html`: all 240 images, browsable, grouped by prompt with prompt text shown

Reviewer-facing supplement. Worth naming in the paper as the artifact-level
disclosure.

### Charts already rendered in `results/*/report.html`

Variance-share bars, the floor curve with bootstrap bands, and the legibility
chart with its chance line, all as self-contained SVG. Screenshot or re-render
these rather than re-plotting; the floor curve for `image_semantic` vs
`text_semantic` on one axis is the strongest quantitative figure available and
does not yet exist as a standalone file. **Making that combined two-modality
floor plot is the one figure worth building before submission.**

### What is missing, figure-wise

There is **no text-side figure**. The text result is currently table-only. If
time allows, the cheapest strong option is a small typographic figure: the same
rung-6 text prompt with four sampled completions side by side, showing that they
obey every stated constraint (three sentences, present tense, ~60 words) and
still say materially different things. That makes the flat semantic floor legible
without a chart.

## 4. Design, as it needs to be stated in a methods section

- **Ladder design.** 6 specificity rungs × 2 prompt families (`landscape`,
  `object`) × models × seeds, per modality. Rungs are **nested**: each strictly
  extends the one below (`"a teapot"` → 34 words), so "more specific" means "more
  constraints added" and nothing else changed. Two families guard against the
  answer being a fact about one subject.
- **Image run:** 2 models (`stable-diffusion-xl-1024-v1-0`,
  `playground-v2-1024px-aesthetic`) × 12 prompts × **10 seeds** = 240 images.
  1024², 30 steps, cfg 7, no negative prompt. Ten seeds per cell rather than the
  design's twenty: balanced and every estimate valid, intervals simply wider.
- **Text run:** 1 model (`gpt-oss-120b`) × 12 prompts × **20 samples** = 240
  generations. temperature 1.0, top_p 1.0, max_tokens 1200.
- **Resampling:** 999 permutations, 500 bootstrap draws, both runs.
- **Four feature views:** `image_semantic` (VLM caption at temperature 0 →
  nomic-embed-text-v1.5), `text_semantic` (same embedder on the generation),
  `image_lowlevel` (56 perceptual stats: hue histogram, tone, colourfulness, edge
  density, 4×4 spatial luminance), `text_lexical` (16 surface stats: length,
  type-token ratio, hapax rate, sentence shape, register). The **semantic pair is
  the cross-modal comparison proper**: captions and generations enter the same
  embedding space, so their variance shares are commensurable. The surface pair
  is a model-free control.
- **Why the ratio is not the headline.** A hand-picked prompt set makes any
  seed/prompt ratio a property of the picker. The nested ladder converts the
  question from "is the seed bigger than the prompt" (unanswerable without a
  prompt distribution) into "how much randomness can specification remove, and
  does it reach zero", which needs no prompt distribution. The floor curve never
  divides by prompt variance, for exactly this reason. Use this as the reply to
  *"your prompt set is arbitrary."*
- **η² vs variance components.** Headline claims rest on EMS-corrected variance
  components, not η². η² inflates prompt and model, since
  E[MS_prompt] = σ²_seed + n·σ²_int + n·b·σ²_prompt. It would systematically
  overstate exactly the thing under scrutiny. Prompt and model are treated as
  random factors.
- **Bootstrap.** Two-level (resampling prompts *and* seeds), log-scale basic
  bootstrap, because resampling prompts with replacement biases the ratio upward.
- **Legibility classifier.** Leave-one-out nearest centroid with a rank-one
  centroid update; without the correction every sample is pulled toward its own
  class and accuracy is optimistic. Runs within a generative model so model
  identity cannot leak in as a cue.

The 12 image prompts, verbatim, are in `results/ladder-image/config.json`; the 12
text prompts in `results/text_ladder_gptoss/config.json`. Both are short enough to
put in an appendix and doing so is worth the space.

## 5. Limitations to state in the paper, not to hide

1. **Fireworks serverless does not honour `seed` on `/chat/completions`.**
   Verified across 10 models: a fixed seed at temperature 1.0 diverges within ~40
   characters on repeat calls, while temperature 0 is deterministic, so the
   parameter is ignored, not lost to batching noise. The text column is
   **sampling variance, not seed variance**. Three of the four analyses never use
   the seed's identity (decomposition, floor, legibility) and survive verbatim
   with the replicate relabelled "run"; the seed-signature test does not, and
   `cli.py` skips it for text rather than printing a meaningless p-value. Do not
   present the two modalities' variance shares under one heading without saying
   this. It is also, conveniently, the paper's sharpest point about agency:
   reproducibility of the residue is a provider decision.
2. **Single text model.** Nine of ten candidates leaked chain-of-thought into
   `content` or returned empty content at the tightest rungs; `gpt-oss-120b` was
   the only one that held. The `model` and `prompt×model` terms read 0.0% as a
   fallback, and there is no second model to check the text floor against.
3. **Ten seeds per image cell, not twenty.** Intervals are wider than a full run's.
4. **Neither floor curve is monotone.** r5 > r4 in both image views; the
   `image_lowlevel` curve *rises* at rungs 2–3 before falling. A plausible reading
   is that `"a landscape"` collapses onto a narrow mode and `"coastal … at dusk"`
   opens up palette and lighting range before later constraints close it down.
   That is a hypothesis, not a result, and 10 seeds per cell cannot resolve it.
   The `text_semantic` non-monotonicity (peaking at 120% of rung 1 at r4) is the
   same caveat and matters more, because that row carries a claim.
5. **The captioner is a lossy bottleneck.** It runs at temperature 0 so it adds no
   variance of its own, but image variation it does not bother to describe is
   invisible to `image_semantic`. That is why `image_lowlevel` exists as a
   control, and the two views disagreeing (seed/prompt 0.50 vs 1.37) is itself a
   finding to report rather than average away.
6. **Distance is not perceived difference.** Two images can sit far apart in
   embedding space and read as the same picture. A large seed share is evidence
   that unexamined randomness does authorial work; it is not proof that the work
   is the *kind* a viewer cares about. The ladder figure is the honest answer here:
   it lets the reader judge the residue directly.
7. **Two prompt families, two subjects.** Everything here is a claim about
   teapots and coastlines.

## 6. Suggested structure, 6 pages

1. **Framing (0.75 p).** The prompt is treated as the locus of authorship. Ask
   what is left underneath it, and whether the artist can reach it. Agency theme
   lands here.
2. **Method (1 p).** Ladder design, four views, ANOVA decomposition, the four
   analyses. Fig. 1 = the ladder figure.
3. **Result 1: the prompt wins (0.5 p).** Report the negative result plainly with
   the decomposition and legibility tables. This buys credibility for everything
   after it.
4. **Result 2: the floor (1.25 p).** Curve for both modalities. Image semantic
   drops to 31% and stops; text semantic never descends. Fig. 2 = combined floor
   plot (to build).
5. **Result 3: the residue is a hand (1.25 p).** Signature test, luminance
   features, Fig. 3 = signature figure.
6. **Result 4: addressable vs not (0.75 p).** The seed-honouring asymmetry, framed
   as infrastructure deciding who gets to use the residue.
7. **Limitations + what an artist does with this (0.5 p).**

## 7. Reproducing any number in this document

```bash
python -m seedstudy analyze --run results/ladder-image --mock        # images
python -m seedstudy analyze --run results/text_ladder_gptoss         # text
python -m seedstudy figures --run runs/ladder                        # figures (needs the PNGs)
```

`--mock` is safe: with `features/*.npz` present the client is constructed but
never called, so no key is needed and nothing is spent. `FINDINGS.md` predates the
text run and describes text as "not run"; §2 above supersedes it.

---

# Method reference

## The design

A fully crossed **prompt × model** grid with **n seeds per cell**, run once per cell.

```
SS_total = SS_prompt + SS_model + SS_prompt:model + SS_within
                                                    ^^^^^^^^^
                                                    the seed
```

Seed is deliberately *not* a main effect. Seed 7 for prompt A has nothing to do
with seed 7 for prompt B: the levels don't mean anything across cells, so a
"seed main effect" would be meaningless. Seed is the **replicate** dimension, and
because the seed is the only thing varying inside a cell, within-cell variance
*is* seed variance. That equivalence is the whole experiment.

Outputs are vectors, so every sum of squares is summed over feature dimensions
(equivalently the trace of the Gower-centred cross-product matrix; with
Euclidean distance the two are the same thing). The design is balanced and fully
crossed, so the decomposition is exactly orthogonal: no ordering ambiguity, no
Type I/II/III argument to have.

### Two summaries, two questions

| | what it answers | caveat |
|---|---|---|
| **η²** | share of variance *in this sample* | inflated for prompt and model, because their mean squares carry seed noise, since E[MS_prompt] = σ²_seed + n·σ²_int + n·b·σ²_prompt |
| **variance component** | EMS-corrected estimate of the *underlying* variance per source | clamped at zero; noisy when a component is near zero |

Prompt and model are treated as **random** factors for the component estimates.
That's the right call for the question: we care what a *typical* prompt induces,
not about these twelve specific prompts.

### The seed-signature test, and why it matters

The decomposition rests on an assumption: that seed 7 for prompt A shares nothing
with seed 7 for prompt B. That assumption is **testable**, because the same seed
list runs against every prompt. Prompt and seed form a complete crossed grid with
one artifact per cell, so the prompt × seed interaction is the error term (the
standard randomised-block layout with no replication):

```
SS_total = SS_prompt + SS_seed + SS_residual
F        = MS_seed / MS_residual
```

Seed labels are permuted independently *within* each prompt, which breaks
cross-prompt consistency while leaving each prompt's own spread untouched.

If a seed effect survives, the seed is not noise; it is a consistent hand applied
on top of whatever was asked for, and "uncredited collaborator" stops being a
metaphor. The test also names the features it acts on, so the effect can be
described rather than only detected. Run per generative model, since seeds mean
different things to different samplers and pooling would let a model effect pose
as a seed effect. **It is vacuous, and is skipped, when the backend ignores the
seed**: arbitrary labels have no cross-prompt consistency to find.

### What's reported

- variance share per source, per view
- `seed / prompt` variance ratio with a 95% two-level bootstrap interval
- permutation p-values for prompt, model, and interaction, each with a restricted
  shuffle so it only breaks the effect being tested. Seed gets none; it's the
  residual stratum, with no labels to shuffle.
- `P(seed pair > prompt pair)`: the probability that two outputs *sharing* a
  prompt are farther apart than two from *different* prompts. At 0.5 the prompt
  separates nothing at all.
- **specificity floor**: seed variance per rung, as a fraction of the three-word
  prompt's, with a bootstrap band. Normalised per feature dimension so views of
  different width are comparable.
- **intent legibility**: leave-one-out nearest-centroid accuracy, per-rung
  accuracy, and the most confusable prompt pairs.
- **seed signature**: η², F, permutation p, and the named features the seed acts on.

## Install

```bash
make venv
export FIREWORKS_API_KEY=...
```

## Use

```bash
make test                                  # 31 tests, offline
make mock                                  # full pipeline on synthetic data, no API calls
make estimate                              # count the API calls the design implies

make ladder                                # image ladder
python -m seedstudy run --config configs/text.json     # text ladder
python -m seedstudy run --config configs/pilot.json    # small real run first
```

Verify per-call latency and whether the backend honours `seed` with the pilot
before committing to a full sweep. If a model ignores the seed, its within-cell
variance won't be zero (sampling is still stochastic), it just won't be *seed*
variance, which quietly breaks the interpretation. **Fireworks serverless chat
does ignore it; this was found the expensive way.**

`generate` and `analyze` split cleanly, and both resume:

```bash
python -m seedstudy generate --config configs/full.json --modality image
python -m seedstudy analyze  --run runs/default
```

Every API response is cached on disk under a hash of its request
(`.cache/fireworks/`), so an interrupted sweep resumes without re-spending and
re-analysis never re-generates. The manifest is append-only and keyed by
`(modality, model, prompt, seed)`; rerunning `generate` fills only the gaps.

### Output

```
config.json           the exact design that was run
manifest.jsonl        one row per artifact
artifacts/<modality>/<model>/<prompt_id>/<seed>.png|txt
captions.jsonl        VLM captions, if enabled
features/*.npz        cached feature matrices
report.html           self-contained; charts, tables, dark mode
results.json          every number, for downstream work
```

## Design choices worth knowing about

**Text temperature must be > 0.** Greedy decoding is deterministic regardless of
seed, so every "seed variance" would be zero by construction rather than by
measurement. Default is 1.0.

**Captioning runs at temperature 0**, so the captioner adds no variance of its
own to the image semantic view. Disable with `--no-captions` for a cheaper run.

**Semantic views are row-normalised** before distances are taken, so squared
Euclidean distance is a monotone function of cosine distance, the right geometry
for embeddings. Surface views are z-scored only.

**Unbalanced designs are refused, not silently analysed.** A partial sweep raises
rather than quietly producing a decomposition whose components aren't estimable.

## Testing

31 tests, all offline. Every statistical claim is checked against data with a
planted answer:

- `test_variance.py`: plants known variance splits and checks recovery
  (prompt-dominant, seed-dominant, pure noise), plus orthogonality, degrees of
  freedom, the single-model fallback, and refusal to analyse an unbalanced grid.
- `test_experiments.py`: plants a cross-prompt seed effect and checks the
  signature test finds it *and* that it stays null when the seed is pure noise;
  checks the top-features report names the two dimensions the effect was planted
  in; checks legibility is high when prompts are separable, at chance when the
  prompt leaves no trace, and not optimistic on noise; checks the floor curve is
  monotone under decay and detects a plateau above zero.
- `test_pipeline.py`: the whole sweep → features → ANOVA → report chain against
  `MockClient`.

## Repository map

```
seedstudy/                 the package (7 modules: config, client, sweep, features,
                           variance, legibility, report) + cli.py
configs/                   pilot.json, text.json, full.json, ladder via --ladder
results/ladder-image/      240-image run: config, manifest, features, results, reports
results/text_ladder_gptoss/ 240-text run: same, plus verbatim artifacts
results/figures/           the paper figures, contact sheets, gallery
FINDINGS.md                image-run narrative (predates the text run)
DEPOSIT.md, .zenodo.json   archival deposit metadata
seedstudy_figures_images.zip                     figures only, 6.8 MB
seedstudy_ladder-image_240img_2026-08-06.zip     the 240 raw PNGs, 295 MB
seedstudy_ladder-image_analysis_2026-08-06.zip   analysis outputs, 8 MB
```
