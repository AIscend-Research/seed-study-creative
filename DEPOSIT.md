# Depositing the artifacts

The image archive is **294.4 MB**, which GitHub hard-rejects — the per-file limit
on push is 100 MB. Git LFS technically fits but the free tier gives 1 GB of
storage and 1 GB of bandwidth per month, so a handful of readers downloading a
294 MB file would exhaust the quota mid-review. Hence Zenodo: no practical size
limit at this scale, a permanent DOI you can cite in the paper, and no dependence
on this repo staying up.

## Two files to upload

| file | size | what it is |
|---|---|---|
| `seedstudy_ladder-image_240img_2026-08-06.zip` | 294.4 MB | all 240 full-resolution PNGs, manifest, captions, cached features, config |
| `seedstudy_ladder-image_analysis_2026-08-06.zip` | 7.7 MB | reports, `results.json`, paper figures, browsable gallery, configs, findings |

Both are gitignored (`*.zip`). Regenerate the second at any time with:

```bash
zip -r seedstudy_ladder-image_analysis_2026-08-06.zip results FINDINGS.md README.md configs
```

## Steps

1. Sign in at <https://zenodo.org> (ORCID or GitHub both work).
2. **New upload** → drag both zips in.
3. Open `.zenodo.json` in this repo and copy the fields across — title, description,
   keywords, license, and the related identifier pointing at the GitHub repo. Zenodo's
   web form does not read `.zenodo.json` (only the GitHub-release integration does),
   so this is a copy-paste.
4. **Fill in the `creators` block.** It is a `TODO` placeholder — I did not invent
   author names. Add ORCIDs if you have them; they make the DOI resolve to your
   profile.
5. **Reserve DOI** before publishing. That gives you the DOI string immediately, so
   you can cite it in the paper you're submitting *now* rather than waiting.
6. Publish. The DOI activates and the record becomes permanent — Zenodo records
   cannot be deleted after publication, only new versions issued.

## Cite it in the paper as

```
[N] <Authors>. Seed study: seed, prompt, and model contributions to generated
    image variance (specificity ladder, 240 images). Zenodo, 2026.
    https://doi.org/10.5281/zenodo.XXXXXXX
```

Once you have the DOI, add it to the repo README and to `.zenodo.json`'s
`related_identifiers` so the two point at each other.

## Licensing — read this before publishing

`.zenodo.json` declares **CC-BY-4.0**, which is the right call for the parts you
authored: the analysis, metadata, captions, feature matrices, and figures.

It does **not** cleanly cover the 240 generated PNGs. Those come from Stable
Diffusion XL 1.0 (CreativeML OpenRAIL-M) and Playground v2 (Playground v2
Community License), each of which carries its own use restrictions that you cannot
license away by declaring CC-BY over the output. The `notes` field in
`.zenodo.json` states this split explicitly.

This is a flag, not legal advice. If the venue or your institution has a position
on redistributing model outputs, check it before publishing — Zenodo records are
permanent.

## A note on the GitHub–Zenodo integration

Zenodo can watch a GitHub repo and archive each release automatically. It archives
a **source zipball of the repository**, not release binaries — so it would capture
the code and `results/` but *not* the 294 MB image archive, which is gitignored.
Useful for versioning the code with its own DOI; not a substitute for the manual
upload above.
