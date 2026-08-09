# Seed study — results

Run: `runs/text_ladder_gptoss` · 240 artifacts

| view | modality | seed | prompt | model | prompt x model | seed/prompt | 95% CI | P(seed pair > prompt pair) |
|---|---|---|---|---|---|---|---|---|
| text_lexical | text | 21.8% | 78.2% | 0.0% | 0.0% | 0.28 | 0.17–0.36 | 0.14 |
| text_semantic | text | 23.7% | 76.3% | 0.0% | 0.0% | 0.31 | 0.23–0.38 | 0.02 |

## Specificity floor (seed variance by rung)

| view | r1 | r2 | r3 | r4 | r5 | r6 | retained at top |
|---|---|---|---|---|---|---|---|
| text_lexical | 100.0% | 96.1% | 32.6% | 49.7% | 55.4% | 50.7% | **50.7%** |
| text_semantic | 100.0% | 102.4% | 103.3% | 120.5% | 108.8% | 86.6% | **86.6%** |

Values are seed variance as a fraction of the three-word prompt's.

## Intent legibility (prompt recovered from artifact)

| view | accuracy | chance | lift | 95% CI |
|---|---|---|---|---|
| text_lexical | 72.5% | 8.3% | 70.0% | 66.7%–78.3% |
| text_semantic | 99.6% | 8.3% | 99.5% | 98.8%–100.0% |

## Reading this

- Seed variance is within-cell variance: prompt and model are fixed inside a cell,
  so the seed is the only thing left that could have moved the output.
- Variance shares are EMS-corrected components, not raw eta-squared; raw eta-squared
  credits the prompt with seed noise carried in its mean square.
- Prompts are resampled in the bootstrap, so the interval reflects the dominant
  uncertainty: which prompts were chosen, not how many seeds were run.
