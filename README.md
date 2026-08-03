# seed-study

How much of a generated artifact is determined by the random seed alone — before
any human intent expressed in the prompt has a chance to matter? And does the
answer look the same for images and for text?

Fix the prompt. Fix the model. Sweep the seed. Decompose the variance. Do it
identically in both modalities so the two numbers can be compared.

Everything runs on [Fireworks](https://fireworks.ai) — image generation, text
generation, VLM captioning, and embeddings all come from the same provider, so
the two modalities are measured through comparable machinery.

## The design

A fully crossed **prompt × model** grid with **n seeds per cell**, run once per cell.

```
SS_total = SS_prompt + SS_model + SS_prompt:model + SS_within
                                                    ^^^^^^^^^
                                                    the seed
```

Seed is deliberately *not* a main effect. Seed 7 for prompt A has nothing to do
with seed 7 for prompt B — the levels don't mean anything across cells, so a
"seed main effect" would be meaningless. Seed is the **replicate** dimension, and
because the seed is the only thing varying inside a cell, within-cell variance
*is* seed variance. That equivalence is the whole experiment.

Outputs are vectors, so every sum of squares is summed over feature dimensions
(equivalently the trace of the Gower-centred cross-product matrix — with
Euclidean distance the two are the same thing). The design is balanced and fully
crossed, so the decomposition is exactly orthogonal: no ordering ambiguity, no
Type I/II/III argument to have.

### Two summaries, two questions

| | what it answers | caveat |
|---|---|---|
| **η²** | share of variance *in this sample* | inflated for prompt and model — their mean squares carry seed noise, since E[MS_prompt] = σ²_seed + n·σ²_int + n·b·σ²_prompt |
| **variance component** | EMS-corrected estimate of the *underlying* variance per source | clamped at zero; noisy when a component is near zero |

The headline claim rests on the **variance components**, not on η². Reporting η²
alone would systematically overstate exactly the thing under scrutiny — how much
work the prompt is doing.

Prompt and model are treated as **random** factors for the component estimates.
That's the right call for the question: we care what a *typical* prompt induces,
not about these six specific prompts.

### Four feature views

No conclusion rests on one way of measuring "how different are two outputs."

| view | modality | what it is |
|---|---|---|
| `image_semantic` | image | VLM caption → text embedding |
| `text_semantic` | text | embedding of the generated text |
| `image_lowlevel` | image | 56 perceptual stats: hue histogram, tone, colourfulness, edge density, 4×4 spatial layout |
| `text_lexical` | text | 16 surface stats: length, type-token ratio, hapax rate, sentence shape, register |

The **semantic pair is the cross-modal comparison proper** — captions and
generations go through the same embedding model into the same space, so their
variance shares are commensurable. The surface pair is a model-free control: if
it agrees, the result isn't an artifact of the embedder. If it disagrees, that
disagreement is the finding.

### What's reported

- variance share per source, per view
- `seed / prompt` variance ratio with a **95% two-level bootstrap** interval
  (resampling prompts *and* seeds — with six prompts, "which prompts did we
  happen to pick" is the dominant uncertainty, not seed count). Taken as a
  log-scale basic bootstrap, because resampling prompts with replacement biases
  the ratio upward.
- permutation p-values for prompt, model, and interaction, each with a restricted
  shuffle so it only breaks the effect being tested. Seed gets none — it's the
  residual stratum, with no labels to shuffle.
- `P(seed pair > prompt pair)`: the probability that two outputs *sharing* a
  prompt are farther apart than two outputs from *different* prompts. At 0.5 the
  prompt separates nothing at all.

## Install

```bash
make venv
export FIREWORKS_API_KEY=...
```

## Use

```bash
make test                                  # 17 tests, offline
make mock                                  # full pipeline on synthetic data, no API calls
make estimate                              # count the API calls the design implies

python -m seedstudy init --output configs/full.json    # write a config to edit
python -m seedstudy run --config configs/pilot.json    # small real run
python -m seedstudy run --config configs/full.json     # 240 images + 240 generations
```

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

A run directory contains:

```
config.json           the exact design that was run
manifest.jsonl        one row per artifact
artifacts/<modality>/<model>/<prompt_id>/<seed>.png|txt
captions.jsonl        VLM captions, if enabled
features/*.npz        cached feature matrices
report.html           self-contained; charts, tables, dark mode
report.md             the summary table
results.json          every number, for downstream work
```

## Design choices worth knowing about

**Text temperature must be > 0.** Greedy decoding is deterministic regardless of
seed, so every "seed variance" would be zero by construction rather than by
measurement. Default is 1.0.

**Captioning runs at temperature 0**, so the captioner adds no variance of its
own to the image semantic view. It still smooths — a VLM caption is a lossy
bottleneck, and image variance the captioner doesn't bother to describe is
invisible to `image_semantic`. That is exactly why `image_lowlevel` exists as a
control. Disable captions with `--no-captions` for a cheaper run.

**Prompt specificity is varied on purpose** (loose → tight, three levels, paired
across modalities). If the seed/prompt balance shifts with specificity, that's
more interesting than any single pooled number.

**Semantic views are row-normalised** before distances are taken, so squared
Euclidean distance is a monotone function of cosine distance — the right geometry
for embeddings. Surface views are z-scored only.

**Unbalanced designs are refused, not silently analysed.** A partial sweep raises
rather than quietly producing a decomposition whose components aren't estimable.

## Testing

`tests/test_variance.py` plants known variance splits and checks the
decomposition recovers them — prompt-dominant, seed-dominant, and pure noise —
plus orthogonality, degrees of freedom, the single-model fallback, and the
refusal to analyse an unbalanced grid. `tests/test_pipeline.py` runs the whole
sweep → features → ANOVA → report chain against `MockClient`, an offline backend
whose outputs are built from controllable prompt/seed/model strengths.

## What this can and can't show

It measures how much output variation the seed accounts for relative to the
prompt, **for the prompts, models, and features chosen**. It does not measure
whether that variation is one a viewer would care about: two images can sit far
apart in embedding space and read as the same picture, or sit close together and
read as entirely different. A large seed share is evidence that unexamined
randomness is doing authorial work; it is not by itself proof that the work is
the *kind* that matters. Prompt selection is the main lever on the result, which
is why prompts are resampled in the bootstrap and why specificity is varied by
design.
