# Seed study: results

Run: `runs/ladder_image` · 240 artifacts

| view | modality | seed | prompt | model | prompt x model | seed/prompt | 95% CI | P(seed pair > prompt pair) |
|---|---|---|---|---|---|---|---|---|
| image_lowlevel | image | 35.5% | 25.9% | 26.3% | 12.3% | 1.37 | 0.84–2.19 | 0.28 |
| image_semantic | image | 30.9% | 62.4% | 1.3% | 5.4% | 0.50 | 0.31–0.67 | 0.12 |

## Specificity floor (seed variance by rung)

| view | r1 | r2 | r3 | r4 | r5 | r6 | retained at top |
|---|---|---|---|---|---|---|---|
| image_lowlevel | 100.0% | 114.7% | 116.7% | 68.0% | 76.3% | 55.9% | **55.9%** |
| image_semantic | 100.0% | 70.0% | 53.8% | 44.4% | 50.4% | 30.9% | **30.9%** |

Values are seed variance as a fraction of the three-word prompt's.

## Intent legibility (prompt recovered from artifact)

| view | accuracy | chance | lift | 95% CI |
|---|---|---|---|---|
| image_lowlevel | 67.9% | 8.3% | 65.0% | 62.1%–73.8% |
| image_semantic | 74.6% | 8.3% | 72.3% | 68.8%–80.4% |

## Seed signature (is the seed a style?)

| view | model | eta2 seed | eta2 prompt | F | p | acts on |
|---|---|---|---|---|---|---|
| image_lowlevel | playground-v2-1024px-aesthetic | 10.1% | 62.6% | 4.08 | 0.001 | lum_cell_4, lum_cell_0, lum_cell_8 |
| image_lowlevel | stable-diffusion-xl-1024-v1-0 | 16.8% | 47.8% | 5.24 | 0.001 | lum_cell_10, lum_cell_7, lum_cell_0 |
| image_semantic | playground-v2-1024px-aesthetic | 2.9% | 71.2% | 1.22 | 0.002 | e661, e77, e424 |
| image_semantic | stable-diffusion-xl-1024-v1-0 | 4.0% | 68.0% | 1.57 | 0.001 | e261, e109, e524 |

## Reading this

- Seed variance is within-cell variance: prompt and model are fixed inside a cell,
  so the seed is the only thing left that could have moved the output.
- Variance shares are EMS-corrected components, not raw eta-squared; raw eta-squared
  credits the prompt with seed noise carried in its mean square.
- Prompts are resampled in the bootstrap, so the interval reflects the dominant
  uncertainty: which prompts were chosen, not how many seeds were run.
