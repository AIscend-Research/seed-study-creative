# Seed study — results

Run: `runs/mock` · 24 artifacts

| view | modality | seed | prompt | model | prompt x model | seed/prompt | 95% CI | P(seed pair > prompt pair) |
|---|---|---|---|---|---|---|---|---|
| image_lowlevel | image | 53.7% | 46.3% | 0.0% | 0.0% | 1.16 | 0.35–inf | 0.24 |
| image_semantic | image | 100.0% | 0.0% | 0.0% | 0.0% | inf | 2.05–inf | 0.52 |
| text_lexical | text | 100.0% | 0.0% | 0.0% | 0.0% | inf | 0.63–inf | 0.55 |
| text_semantic | text | 100.0% | 0.0% | 0.0% | 0.0% | inf | 2.35–inf | 0.64 |

## Cross-modal (shared embedding space)

- image seed share: 100.0%
- text seed share: 100.0%
- gap: 0.0%

## Reading this

- Seed variance is within-cell variance: prompt and model are fixed inside a cell,
  so the seed is the only thing left that could have moved the output.
- Variance shares are EMS-corrected components, not raw eta-squared; raw eta-squared
  credits the prompt with seed noise carried in its mean square.
- Prompts are resampled in the bootstrap, so the interval reflects the dominant
  uncertainty: which prompts were chosen, not how many seeds were run.
