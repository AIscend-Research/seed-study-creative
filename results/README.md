# Results

Committed run outputs. The generated artifacts themselves (PNGs, ~295MB) stay out
of the repo under `runs/`, which is gitignored. Everything needed to reproduce or
re-analyse the numbers is here: the cached feature matrices mean the analysis
re-runs with **no API calls**.

```bash
unzip ladder_image_results.zip -d runs/ladder      # if you need the images back
python -m seedstudy analyze --run results/ladder-image --mock
```

`--mock` is safe here: with `features/*.npz` present the client is constructed but
never called, so no key is needed and nothing is spent.

## ladder-image (2026-08-06)

The specificity ladder, image modality only.

| | |
|---|---|
| design | 6 rungs × 2 families × 2 models × **10 seeds** |
| artifacts | 240 images, balanced across 24 cells |
| models | `stable-diffusion-xl-1024-v1-0`, `playground-v2-1024px-aesthetic` |
| views | `image_lowlevel` (56 perceptual stats), `image_semantic` (VLM caption → 768-d embedding) |
| resampling | 999 permutations, 500 bootstrap draws |

Ten seeds per cell, not the twenty the design specifies. The decomposition is
still balanced and every estimate is valid; the intervals are simply wider than a
full run's would be.

### Headline numbers

| | `image_lowlevel` | `image_semantic` |
|---|---|---|
| variance from seed | 35.5% | 30.9% |
| variance from prompt | 25.9% | 62.4% |
| seed/prompt ratio | 1.37 (CI 0.84–2.19) | 0.50 (CI 0.31–0.67) |
| seed variance retained at rung 6 | 55.9% | 30.9% |
| prompt recovered from artifact | 67.9% (chance 8.3%) | 74.6% (chance 8.3%) |
| seed signature | p = 0.001, both models | p ≤ 0.002, both models |

See `../FINDINGS.md` for what these mean and where they cut against the study's
original hypothesis.

## text

Not yet run. `configs/text.json` is ready and points at a model that resolves on
the current Fireworks catalog, but see `../FINDINGS.md` — Fireworks serverless does
not honour `seed` on `/chat/completions`, so the text replicate is an
uncontrolled resample rather than a seed. Three of the four analyses survive that;
the seed-signature test does not.
