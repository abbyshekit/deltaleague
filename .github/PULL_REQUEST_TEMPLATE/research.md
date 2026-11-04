<!-- Research PR. Files go under research/<github-user>-<slug>/ (see research/README.md). Fill every section; write "none" where that is the answer. Never include credentials, private test data, weights, signed URLs or generation files. -->

## Summary
- Model competition: `qwen35-08b-v1`
- Parent version id:
- Evaluation config id:
- Submission title (as given to `delta submit --title`):
- One sentence: what changed in the weights and the claimed effect.

## Hypothesis
Why this edit should raise the score, in two or three sentences. Name the behaviour it targets (runaway generation, answer format, arithmetic, recall).

## Edit
- Format: sparse diff / dense diff / LoRA / full checkpoint
- Tensors touched and changed entries:
- Editor: human / agent (model, tool, version)
- Search or fitting procedure, and what it could see (the public validation items are visible to the editor; say whether it used them):
- Seeds:

## Controls
- Against the parent (`delta compare parent.json edit.json`): diff, 95% CI, sign-test p, truncation rate before → after.
- Against the decoding-tuned parent (`delta grid` on the parent, then `delta compare grid-results/<best>.json edit.json`): best cell, diff, 95% CI.
- Scripted baseline: the simplest fixed-rule edit tried without search, and its score (or "none").

## Reproduction
Commands from a fresh `delta competition init`, the edit script in this directory, and the expected SHA-256 of each changed tensor after the edit.

## Submission (after `delta submit`)
- Submission id:
- Version id:
- Official validation / private test score:

## Checklist
- [ ] No credentials, tokens or signed URLs
- [ ] No private test examples or generations
- [ ] No weight files (`*.safetensors`, `*.npz`, `*.pt`) and no `*.generations.jsonl`
- [ ] Parent and evaluation ids match `delta-workspace.json`
- [ ] `delta validate` passed on the submitted directory
