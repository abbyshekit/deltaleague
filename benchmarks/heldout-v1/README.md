# Held-out set v1: pipeline and seal (competition proposal)

The pipeline that built a held-out test set for proposal 3 of `PROTOCOL.md`, with the sealed hash of the result. **The items themselves are not in this repository**: a held-out set that is public is not held out; hidden evaluation material belongs in the private platform repository (CONTRIBUTING.md) with the platform's data on R2. The set (`heldout-v1.json`, 313 items in the `validation.json` format, SHA-256 in `heldout-v1.sha256`) and its private sidecar (solutions, programs, solver answers, option rotations) are delivered to the maintainer out of band; `report.json` carries the counts.

Exposure note: an earlier revision of this branch carried the items for about an hour on a public fork before this rewrite. Treat v1 as exposed for ranking purposes: run the pipeline again for a fresh set before using one on the private test, and keep v1 as a development set. Three families mirror the public validation set's format and scorers: `arc_fresh` (science multiple choice, scorer `choice`), `gsm_fresh` (grade-school word problems, scorer `math`), `math_fresh` (competition problems, scorer `math`). Every prompt ends with the platform's own instruction line. A delivered file runs locally with `delta evaluate --validation heldout-v1.json` (#10), which records its hash in the report.

## How it was built
1. Writers (Claude Opus) wrote ten original items per batch for an assigned topic, with a worked solution and, for `gsm_fresh`, a Python `solve()` that computes the answer from the stated quantities.
2. Two solvers (Claude Sonnet) solved each item blind; the second was told to hunt for ambiguity, a second defensible answer, or a missing quantity.
3. A critic (Claude Opus) rejected items that are a near-duplicate of a specific item it could name from ARC, GSM8K, MATH, MMLU, GPQA, AQuA, SVAMP, ASDiv, MathQA, AIME/AMC or a well-known textbook, items with more than one defensible answer, and items guessable from surface cues. In the first pass the critic also rejected items for sharing a *template* with public items (seasons, frost wedging, ratio changes); that rule was dropped for the second pass, because the public benchmarks are built from the same templates and a held-out set must exclude memorized items, not topics.
4. An item was accepted only when the writer and both solvers agreed on the answer (a Sonnet judge decided equivalence when two LaTeX forms differed) and the critic kept it.
5. Deterministic checks then vetoed, with no model in the loop: a `gsm_fresh` program whose result differs from the stated answer; a `math_fresh` or `gsm_fresh` solver answer that `math-verify` (the platform's scorer) does not accept against the stated answer; any 10-word overlap with any item of the public ARC-Challenge test (1,172), GSM8K test (1,319) or MATH-500 (500) sets; duplicate bodies.
6. `arc_fresh` options were rotated deterministically (item index mod 4) so the correct letter is balanced; the option text is unchanged.
7. The file is sealed by its SHA-256 (`heldout-v1.sha256`, committed). Solutions, programs and solver answers are in `heldout-v1.private.json`, delivered with the set and never shown to a model under evaluation.

## Counts
Two passes of the workflow (pass 1 with the template rule, pass 2 without), merged and sealed together:

| Family | Written | Accepted by the agents | Rejected by the agents | Vetoed by the checks | Sealed |
|---|---|---|---|---|---|
| `arc_fresh` | 420 | 107 | 302 | 0 | **107** |
| `gsm_fresh` | 120 | 100 | 11 | 0 | **100** |
| `math_fresh` | 360 | 106 | 236 | 0 | **106** |

313 items in all. `arc_fresh` correct letters before rotation {'A': 34, 'B': 26, 'C': 24, 'D': 23}, after {'A': 27, 'B': 27, 'C': 27, 'D': 26}. Public sets scanned for 10-gram overlap: ARC-Challenge test 1172, GSM8K test 1319, MATH-500 500; no sealed item shares a 10-word sequence with any of them (the same check vetoed a planted public item in the dry run). Agent-side rejections, with the reason for each, are delivered with the set, not committed: they contain item text. SHA-256 of `heldout-v1.json`: `1c94d96e688c15339c86ff37e0fe3d809c434a5dca650c4324858a8c3a7cec4a`.

Hand check: 16 items re-solved by hand across the two passes (nine from pass 1, seven from pass 2, including the solver-disagreement-prone families); all 16 answers correct.

## What this set is and is not
- Items are new text: no 10-gram of any item appears in the three public test sets, and the critic could not name a source item. They were written by Claude models in November 2026, so they post-date the public sets; whether they post-date a given base model's training cutoff is a property of that model, not of the set.
- The set measures the same skills in the same format as the public validation set, at a similar difficulty by construction, not by calibration against human raters.
- Scores on it are advisory until the evaluator adopts it; `delta compare` pairs two local runs on it by its hash.

## Reproduce
```
python fetch_public.py
python build_heldout.py <workflow results...> <out_dir>
```
`fetch_public.py` downloads the three public test splits into `public/` and prints each row count and SHA-256. `build_heldout.py` takes the result JSONs the two workflow scripts (`workflow-pass1.js`, `workflow-pass2.js`) produce, applies the deterministic vetoes and writes the sealed set, the private file, the veto and rejection records, `heldout-v1.sha256` and `report.json` into `<out_dir>`.

The committed set was built from runs `wf_616c5f47-fbe` (pass 1) and `wf_7a2c8833-e67` (pass 2). Their result JSONs, and the per-item rejection log, are delivered with the set rather than committed, because they contain item text.

The build executes model-written Python (the `gsm_fresh` `solve()` programs). Each program is first parsed with `ast` and vetoed if it imports anything but `math`, calls `open`, `exec`, `eval`, `__import__`, `compile`, `input` or `breakpoint`, or touches an attribute of `os`, `sys`, `subprocess`, `socket` or `builtins`. It then runs under `python -I` in an empty temporary directory with `PATH=/usr/bin:/bin` as the only environment variable, a 10 s timeout, and `setrlimit` caps of 512 MB address space, 5 s CPU, 1 MB written file size and 16 open files; captured output is truncated to 4 KB. macOS refuses `RLIMIT_AS`; the build skips any limit the OS rejects. This is a screen and resource limit, not an isolation boundary: run the build in a disposable environment if the workflow output is not yours.
