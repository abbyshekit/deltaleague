# Final-norm scale: a weight edit that is only a decoding change

## Summary
- Model competition: `qwen35-08b-v1`
- Parent version id (`parent_version_id` in `delta-workspace.json`): `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`
- Evaluation config id (`config_id`): `runtime-v5`
- Submission title (as given to `delta submit --title`): `Final norm × 1.667 (temperature-0.6 equivalent)`
- One sentence: every entry of `model.language_model.norm.weight` is multiplied by α = 1/0.6; nothing else changes, and under the fixed server decoding this equals sampling at temperature 0.6 with the presence penalty scaled to 0.9.

## Hypothesis
The published edit's day-one validation gain (two ARC items) arrives with truncation falling from 22 to 14 of 39 generations. A sharper output distribution can cut runaway thinking without any change in what the model knows. Qwen's published thinking-mode recommendation for Qwen3 is temperature 0.6, top-p 0.95, top-k 20; the server samples Qwen3.5 at temperature 1.0. Scaling the final RMSNorm by α multiplies every logit by α, which is temperature 1/α; because vLLM subtracts the presence penalty from the logits before the temperature division, the penalty's effective size also falls to 1.5/α. If this search-free edit lands within the tie band of the published edit on validation and matches it on the private test, the leaderboard is rewarding decoding, not weight-space reasoning gains, and an "Original (decoding-tuned)" reference row is needed.

Pre-registered predictions, written before any server score:
1. Truncation rate below the parent's 56.4% of 39 generations.
2. Public validation within 25 points of the parent (40.00) and of the published edit (46.67); the tie band at n = 30.
3. Private test unchanged at 33.33 or within one item (22.22–44.44); n = 9.
4. Locally, with the same seeds, the patch and the decoding cell `temperature 0.6, presence_penalty 0.9` give the same scores and near-identical generations (bfloat16 rounding of α·w aside).

## Edit
- Format: sparse diff
- Tensors touched and changed entries: `model.language_model.norm.weight`, up to 1,024 entries (see `results/manifest.json`)
- Editor: human, by a fixed rule; no search, no agent
- Search or fitting procedure, and what it could see: none; α = 1/0.6 was chosen from Qwen3's published decoding recommendation before any evaluation
- Seeds: server seeds; local runs use the CLI's per-sample seeds (1000 + sample)

## Controls
Needs the CLI with `delta compare` and `delta grid` (#1, #2).
- Against the parent (`delta compare`): filled after the run.
- Against the decoding-tuned parent (`delta grid --grid grid-alpha.json` with the cell `temp-0.6-pp-0.9`, then `delta compare`): filled after the run.
- Scripted baseline: this edit is the scripted baseline; the published edit's reproduction (`examples/blog-reproduction`) is the comparison arm.

## Reproduction
From a fresh `delta competition init qwen35-08b-v1 --output experiment`, inside `experiment/`:
```sh
python scale_norm.py 1.6666667 edited-alpha.safetensors
delta patch build --parent model --modified edited-alpha.safetensors --output submission-alpha
delta validate submission-alpha
```
Expected: the manifest lists one tensor, `model.language_model.norm.weight`; its SHA-256 is in `results/manifest.json`.

## Submission (after `delta submit`)
- Submission id:
- Version id (from `delta leaderboard --model qwen35-08b-v1`):
- Official public validation / private test score:

## Checklist
- [x] No credentials, tokens or signed URLs
- [x] No private test examples or generations
- [x] No weight files (`*.safetensors`, `*.npz`, `*.pt`) and no `*.generations.jsonl`
- [x] Parent and evaluation ids match `delta-workspace.json`
- [x] `delta validate` passed on the submitted directory
