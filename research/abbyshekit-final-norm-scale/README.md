# Final-norm scale: a weight edit that is only a decoding change

## Summary
- Model competition: `qwen35-08b-v1`
- Parent version id (`parent_version_id` in `delta-workspace.json`): `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`
- Evaluation config id (`config_id`): `runtime-v5`
- Submission title (as given to `delta submit --title`): `Final norm gain × 1.667 (temperature-0.6 equivalent)`
- One sentence: the gain of `model.language_model.norm.weight` is multiplied by α = 1/0.6 and nothing else changes; because the logits are linear in that gain, the edit equals sampling the parent at temperature 0.6 with the presence penalty at 0.9 under the server's fixed decoding.

## Hypothesis
The server decodes at temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, thinking on. That is the setting the Qwen3.5 model card recommends for thinking-mode text tasks (model card, "Best practices"). The same card reports the model's own benchmark scores at temperature 0.6 in thinking mode ("all other benchmarks adopted the same sampling configuration but with temperature=0.6 under the thinking mode"). The published edit's day-one validation gain (two ARC items) arrives with truncation falling from 22 to 14 of 39 generations, and a sharper distribution can shorten thinking traces without changing what the model knows. If a search-free edit that only reproduces temperature 0.6 lands within the tie band of the published edit on public validation and matches it on the private test, the board is rewarding decoding, and an "Original (decoding-tuned)" reference row is needed.

Mechanism. Qwen3.5's RMSNorm is zero-centred: the output is `norm(h) · (1 + w)`, with `w` stored. The language-model head is the tied embedding with no bias and no logit soft-capping, so the logits are linear in the gain: scaling `(1 + w)` by α scales every logit by α. The stored weight therefore becomes `α·(1 + w) − 1` (computed in float32, one cast to bfloat16; about 0.2% rms rounding error per entry, deterministic). With vLLM's v1 sampler order (penalties subtracted from the logits, then the temperature division, then top-k and top-p), the edit equals temperature 1/α with the presence penalty's effective size at 1.5/α = 0.9. That ordering is assumed for the server and is not verified against it; if the server applied the penalty after the temperature, the edit would behave as temperature 0.6 with the full penalty of 1.5.

Pre-registered predictions, written before any server score:
1. Truncation rate below the parent's 56.4% of 39 generations.
2. Public validation within 25 points of the parent (40.00) and of the published edit (46.67): the worst-case unpaired 95% band at n = 30.
3. Private test unchanged at 33.33 or within one item (22.22–44.44); n = 9, so one item moves the score 11.11 points.
4. Locally, next-token logits of the patched model equal α times the parent's logits up to bfloat16 rounding (`check_logits.py`), and the patch and the decoding cell `temperature 0.6, presence_penalty 0.9, top_p 0.95, top_k 20, thinking on` give the same scores and truncation counts under the same seeds. Bit-identical texts are not expected once a sampled token flips.

These predictions are weak at n = 30 and n = 9; the experiment cannot separate small effects. It can show whether a pure decoding edit is indistinguishable from the published edit on this board.

Second arm, pre-registered before the first arm's score was known (submitted while the first was still queued): α = 1.25, the temperature-0.8 twin (presence penalty 1.2), stored weight `1.25·(1 + w) − 1`, `results/manifest-alpha-1.25.json`. Predictions: truncation between the parent's 56.4% and arm 1's; validation within the same 25-point band; private test within one item. Together the two arms give a dose-response pair for the decoding explanation.

## Edit
- Format: sparse diff
- Tensors touched and changed entries: `model.language_model.norm.weight`, 1,024 entries (`results/manifest.json`)
- Editor: human, by a fixed rule; no search, no agent
- Search or fitting procedure, and what it could see: none; α = 1/0.6 was taken from the temperature at which the model card reports the model's thinking-mode benchmarks, before any evaluation
- Seeds: server seeds; local runs use the CLI's per-sample seeds (1000 + sample)

## Controls
Needs the CLI with `delta compare` and `delta grid` (#1, #2).
- Logit linearity (`check_logits.py 1.6666667`, float32 on CPU, three prompts, last-position logits): max |edited − α·base| = 0.028–0.030 on logits of magnitude 24.6–25.2 (about 0.1%); median ratio edited/base 1.6676–1.6677 against the target 1.6667; argmax unchanged on all three. Effective per-entry gain after the bfloat16 cast: 1.6632–1.6716.
- Against the parent (`delta compare results/grid-default.json results/alpha-patch.json`; MPS, float32, 2 items per benchmark, 1,024 output tokens, thinking on, 6 generations per run): parent 16.7, patch 16.7, difference 0.0 with CI [0, 0]; both truncate 6 of 6 generations at this budget. Advisory only: every thinking trace exceeds 1,024 tokens, so the quick set measures truncation, not answers.
- Against the decoding twin (`delta grid --grid grid-alpha.json`, cell `temp-0.6-pp-0.9`, then `delta compare`): twin 33.3 with 5 of 6 truncated, patch 16.7 with 6 of 6; difference −16.7, CI [−33.3, 0], sign test p = 1.0, not separable. Under the same seeds 3 of the 6 generations are identical character for character and the other 3 diverge after 631, 1,114 and 1,247 characters: bfloat16 rounding of α·(1 + w) perturbs the logits by about 0.1%, and one flipped token changes the rest of the sample. Prediction 4's "same scores" was too strong at this budget; the logit-level check above is the equivalence evidence.
- Scripted baseline: this edit is the scripted baseline; the published edit's reproduction (`examples/blog-reproduction`) is the comparison arm. At the same local budget it scores 0.0 with 6 of 6 truncated (`results/blog-edit.json`, `results/compare-parent-blog.json`).

## Reproduction
From a fresh `delta competition init qwen35-08b-v1 --output experiment`, inside `experiment/`:
```sh
python scale_norm.py 1.6666667 edited-alpha.safetensors
python check_logits.py 1.6666667
delta patch build --parent model --modified edited-alpha.safetensors --output submission-alpha
delta validate submission-alpha
```
Expected: the manifest lists one tensor, `model.language_model.norm.weight`; the artifact SHA-256 is in `results/manifest.json`.

## Submission (after `delta submit`)
- Submission id, arm 1 (α = 1.667): `8ff0f88e-249e-43ea-9fbd-1668b592ecdb` (submitted 2026-11-04T18:04Z)
- Submission id, arm 2 (α = 1.25): `f7742cff-f016-4ba2-bc04-e734086a331d` (submitted 2026-11-04T18:39Z while arm 1 was still queued)
- Version id (from `delta leaderboard --model qwen35-08b-v1`):
- Official public validation / private test score:

## Checklist
- [x] No credentials, tokens or signed URLs
- [x] No private test examples or generations
- [x] No weight files (`*.safetensors`, `*.npz`, `*.pt`) and no `*.generations.jsonl`
- [x] Parent and evaluation ids match `delta-workspace.json`
- [x] `delta validate` passed on the submitted directory
