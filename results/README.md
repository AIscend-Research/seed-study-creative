# Results

Committed run outputs. `runs/` is gitignored working space; anything meant to be
read by someone else gets copied here.

## text_ladder_gptoss

Text modality, specificity ladder, 240 artifacts. One generative model.

| view | sampling | prompt | ratio | 95% CI |
|---|---|---|---|---|
| text_lexical | 21.8% | 78.2% | 0.28 | 0.17–0.36 |
| text_semantic | 23.7% | 76.3% | 0.31 | 0.23–0.38 |

Specificity floor, as a fraction of the three-word prompt's within-cell variance:

| view | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|
| text_lexical | 100.0% | 96.1% | 32.6% | 49.7% | 55.4% | 50.7% |
| text_semantic | 100.0% | 102.4% | 103.3% | 120.5% | 108.8% | 86.6% |

Intent legibility: 99.6% semantic, 72.5% lexical, against 8.3% chance.

The semantic floor is the result worth arguing about. Six rungs of added
constraint, ending at 35 words of pinned-down detail, leave semantic within-cell
variance roughly where three words left it. The lexical floor halves over the
same range, so the constraints do regularise length and sentence shape while the
semantic spread stays put. One model, so treat it as a hypothesis.

### Three things that change how these numbers read

**The column is sampling variance, not seed variance.** Fireworks serverless does
not honour `seed` on `/chat/completions`. Verified across 10 models: a fixed seed
at temperature 1.0 diverges within ~40 characters on repeat calls, while
temperature 0 is deterministic, so the parameter is ignored rather than lost to
batching noise. Within-cell variance is real and it is what these numbers
measure, but no generation here is reproducible and nothing indexed by seed
number means anything. The seed-signature experiment is skipped for text for this
reason (`cli.py`), not because it returned null.

**Single model.** Nine of ten candidates leaked chain of thought into `content`
or returned empty content at the tightest rungs. `gpt-oss-120b` was the only one
that held. The `model` and `prompt × model` terms therefore read 0.0%, which is
the single-model fallback rather than a finding, and there is no second model to
check the floor against.

**Not comparable to the image side yet.** Diffusion models honour seeds and these
text models do not, so when both halves exist the two variance shares are not
measuring the same thing and should not be presented under one heading without
saying so.

### Contents

```
config.json      exact design that ran
manifest.jsonl   one row per artifact
artifacts/       all 240 generations, verbatim
report.md        summary table
report.html      charts and full tables, self-contained
results.json     every number
```

`features/*.npz` is omitted as regenerable cache. To rebuild the analysis from
the committed artifacts:

```bash
cp -R results/text_ladder_gptoss runs/ && python -m seedstudy analyze --run runs/text_ladder_gptoss
```

Generated 2026-08-09 with `configs/text.json` (ladder design, 12 prompts, 20
seeds, temperature 1.0, max_tokens 1200).
