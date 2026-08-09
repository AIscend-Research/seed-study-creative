# Findings — image ladder, 2026-08-06

240 images, 6 specificity rungs × 2 families × 2 models × 10 seeds.

## The original hypothesis did not survive, and what replaced it is better

The study set out to show that seed variance "rivals or exceeds" prompt variance,
so that the prompt is doing less authorial work than it's credited with. In the
semantic view that is **false, and decisively so**:

| | seed | prompt | seed/prompt (95% CI) |
|---|---|---|---|
| `image_semantic` | 30.9% | **62.4%** | 0.50 (0.31–0.67) |
| `image_lowlevel` | **35.5%** | 25.9% | 1.37 (0.84–2.19) |

The semantic interval excludes 1.0 — the prompt wins, and not narrowly. The
low-level interval straddles 1.0, so seed dominance isn't established there
either. Two more results point the same way:

- **Intent is highly recoverable.** A classifier identifies which of 12 prompts
  produced an image **74.6%** of the time against 8.3% chance. Whatever the seed
  is doing, it is not erasing the specification.
- `P(seed pair > prompt pair)` is **0.12** semantically. Two images sharing a
  prompt are far more alike than two from different prompts.

Reporting "seed beats prompt" from this data would require ignoring the strongest
evidence in it. The interesting claim is elsewhere.

## The seed is a hand, not noise — and it works on composition

The seed-signature test is **significant on every view and both models**
(p = 0.001–0.002). Seed labels were permuted independently within each prompt, so
this is cross-prompt consistency: seed 7 does something recognisable *regardless
of what was asked for*.

What it does is spatial:

| model | η² seed | features the seed acts on |
|---|---|---|
| `stable-diffusion-xl` | 16.8% | `lum_cell_10` (.61), `lum_cell_7` (.50), `lum_cell_0` (.49) |
| `playground-v2` | 10.1% | `lum_cell_4` (.48), `lum_cell_0` (.36), `lum_cell_8` (.31) |

Every top feature is a **spatial luminance cell** — where light falls in the
frame. The seed does not choose what you get; it chooses how it is arranged, and
it makes the same choice every time you reuse it.

Note the asymmetry: the signature is strong low-level (η² 10–17%) and weak
semantic (η² 3–4%, significant but small). The seed's consistent hand is
**compositional, not semantic**. It is a lighting and layout collaborator.

This is mechanistically unsurprising — the initial latent is what the seed sets,
and it most directly constrains global structure. Being expected does not make it
uninteresting: the finding is that the constraint is *stable across every prompt*,
which is what makes it a style rather than a perturbation.

## Specification has a floor

Seed variance as a fraction of the three-word prompt's:

| view | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|
| `image_semantic` | 100% | 70.0% | 53.8% | 44.4% | 50.4% | **30.9%** |
| `image_lowlevel` | 100% | 114.7% | 116.7% | 68.0% | 76.3% | **55.9%** |

Going from three words to thirty-five removes about **69%** of the semantic
randomness. The remaining ~31% is what specification cannot buy back — and the
bootstrap band at rung 6 (0.123–0.199) sits well clear of zero.

**Two honest wrinkles.** The low-level curve *rises* at rungs 2–3 before falling:
adding the first few words increased pixel-level variance. A plausible reading is
that `"a landscape"` collapses onto a narrow mode, and `"coastal … at dusk"` opens
up palette and lighting range before later constraints close it down again. That's
a hypothesis, not a result. And the decline is not monotone in either view (r5 > r4
in both), which 10 seeds per cell is too few to resolve.

## Text: not run, and the design needs an amendment

No text artifacts exist yet. Two blockers were found:

1. The originally configured text models 404 on the current Fireworks catalog.
   `configs/text.json` fixes this.
2. **Fireworks serverless does not honour `seed` on `/chat/completions`.** Across
   10 models, a fixed seed at temperature 1.0 diverges within ~40 characters on
   repeat calls; at temperature 0 those models are deterministic, so the parameter
   is being ignored rather than lost to batching noise.

The second is the one that matters, and it is narrower than it looks. Three of the
four analyses only need **repeated samples at a fixed prompt** — they never use the
seed's identity:

| analysis | needs seed labels? | survives? |
|---|---|---|
| variance decomposition | no | yes — relabel the replicate "run", not "seed" |
| specificity floor | no | yes |
| intent legibility | no | yes |
| seed signature | **yes** | no — vacuous when labels are arbitrary |

So the cross-modal comparison is still available for the claim it was built for,
provided the text side is described as *sampling* variance rather than *seed*
variance. That is a real difference and it should be stated plainly rather than
smoothed over: for images the randomness is addressable and reproducible; for text
on this provider it is neither. `cmd_analyze` skips the signature test on text
views rather than reporting a p-value that means nothing.

If the seed signature matters for text — and it is the paper's best finding on the
image side — it needs a backend that honours seeds. A local runner (llama.cpp,
vLLM) does; that is a day of work, not an afternoon.
