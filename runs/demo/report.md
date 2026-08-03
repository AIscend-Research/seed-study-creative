# Seed study — results

Run: `runs/demo (synthetic illustration)` · 960 artifacts

| view | modality | seed | prompt | model | prompt x model | seed/prompt | 95% CI | P(seed pair > prompt pair) |
|---|---|---|---|---|---|---|---|---|
| image_lowlevel | image | 50.9% | 43.2% | 5.9% | 0.0% | 1.18 | 1.13–1.81 | 0.02 |
| image_semantic | image | 64.9% | 27.1% | 7.9% | 0.0% | 2.39 | 2.21–4.23 | 0.12 |
| text_lexical | text | 50.6% | 43.9% | 5.4% | 0.0% | 1.15 | 1.12–1.90 | 0.02 |
| text_semantic | text | 25.5% | 65.8% | 8.6% | 0.0% | 0.39 | 0.37–0.63 | 0.00 |

## Cross-modal (shared embedding space)

- image seed share: 64.9%
- text seed share: 25.5%
- gap: 39.4%

## Reading this

- Seed variance is within-cell variance: prompt and model are fixed inside a cell,
  so the seed is the only thing left that could have moved the output.
- Variance shares are EMS-corrected components, not raw eta-squared; raw eta-squared
  credits the prompt with seed noise carried in its mean square.
- Prompts are resampled in the bootstrap, so the interval reflects the dominant
  uncertainty: which prompts were chosen, not how many seeds were run.
