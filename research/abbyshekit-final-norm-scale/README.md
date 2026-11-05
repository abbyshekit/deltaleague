# Final-norm scale: a weight edit equivalent to rescaling every logit

## Summary
- Model competition: `qwen35-08b-v1`
- Parent version id (`parent_version_id` in `delta-workspace.json`): `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`
- Evaluation config id (`config_id`): `runtime-v5` (submitted and verified there; the board re-scored both arms under `public-ladder500-8k-v2` on 2026-11-04, see the last section)
- Submission title (as given to `delta submit --title`): `Final norm gain × 1.667 (temperature-0.6 equivalent)`
- One sentence: the gain of `model.language_model.norm.weight` is multiplied by α = 1/0.6 and nothing else changes; because the logits are linear in that gain, the edit scales every logit by α, which equals sampling the parent at temperature 0.6 with the presence penalty at 0.9 if the server applies penalties before the temperature division (assumed, not verified).

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
- Logit linearity (`check_logits.py 1.6666667`, float32 on CPU, three prompts, last-position logits): max |edited − α·base| = 0.028–0.030 on logits of magnitude 24.6–25.2, a relative deviation of 0.11–0.12%; ratio edited/base over the 181k–187k vocabulary entries with |base| > 1: median 1.6676–1.6677, range 1.642–1.693 against the target 1.6667; argmax unchanged on all three. The script exits non-zero above 1% deviation or on an argmax change. Arm 2 (`check_logits.py 1.25`, same prompts): max |edited − α·base| = 0.025–0.028 on logits of magnitude 24.6–25.2, relative deviation 0.10–0.11%; ratio median 1.2505–1.2507, range 1.231–1.271; argmax unchanged on all three. Effective per-entry gain after the bfloat16 cast: 1.6632–1.6716.
- Against the parent (`delta compare results/grid-default.json results/alpha-patch.json`; MPS, float32, 2 items per benchmark, 1,024 output tokens, thinking on, 6 generations per run): parent 16.7, patch 16.7, difference 0.0 with CI [0, 0]; both truncate 6 of 6 generations at this budget. Advisory only: every thinking trace exceeds 1,024 tokens, so the quick set measures truncation, not answers.
- Against the decoding twin (`delta grid --grid grid-alpha.json`, cell `temp-0.6-pp-0.9`, then `delta compare`): twin 33.3 with 5 of 6 truncated, patch 16.7 with 6 of 6; difference −16.7, CI [−33.3, 0], sign test p = 1.0, not separable. Under the same seeds 3 of the 6 generations are identical character for character and the other 3 diverge after 631, 1,114 and 1,247 characters: bfloat16 rounding of α·(1 + w) perturbs the logits by about 0.1%, and one flipped token changes the rest of the sample. Prediction 4's "same scores" was too strong at this budget; the logit-level check above is the equivalence evidence.
- Scripted baseline: this edit is the scripted baseline; the published edit's reproduction (`examples/blog-reproduction`) is the comparison arm. At the same local budget it scores 0.0 with 6 of 6 truncated (`results/blog-edit.json`, `results/compare-parent-blog.json`).

## Reproduction
From a fresh `delta competition init qwen35-08b-v1 --output experiment`, copy this directory's scripts into the workspace, then work inside `experiment/`:
```bash
cp /path/to/deltaleague/research/abbyshekit-final-norm-scale/{scale_norm.py,check_logits.py,grid-alpha.json} experiment/
cd experiment
python scale_norm.py 1.6666667 edited-alpha.safetensors      # arm 1; arm 2: python scale_norm.py 1.25 edited-alpha125.safetensors
python check_logits.py 1.6666667                             # arm 2: 1.25
delta patch build --parent model --modified edited-alpha.safetensors --output submission-alpha
delta validate submission-alpha
```
Expected: the manifest lists one tensor, `model.language_model.norm.weight`; the artifact SHA-256 is in `results/manifest.json`.

## Submission (after `delta submit`)
- Submission id, arm 1 (α = 1.667): `8ff0f88e-249e-43ea-9fbd-1668b592ecdb` (submitted 2026-11-04T18:04Z)
- Submission id, arm 2 (α = 1.25): `f7742cff-f016-4ba2-bc04-e734086a331d` (submitted 2026-11-04T18:39Z while arm 1 was still queued)
- Version id (from `delta leaderboard --model qwen35-08b-v1`):
- Official public validation / private test score, arm 1 (α = 1.667): **13.33 / 0.00** (ARC 10, GSM8K 20, MATH-500 10; truncation 92.3% = 36 of 39; 1,196,452 output tokens; 801 generation seconds on an RTX 5090; verified 2026-11-04T20:05Z; version `d17ce35f-90e7-4582-85b5-4659f4a7704b`; `results/submission-arm1.json`)
- Official public validation / private test score, arm 2 (α = 1.25): **23.33 / 22.22** (ARC 30, GSM8K 30, MATH-500 10; truncation 74.4% = 29 of 39; 1,056,680 output tokens; 619 generation seconds; verified 2026-11-04T20:17Z; version `4fe2f3d6-6487-428d-99be-7ce9cf8a2cdb`; `results/submission-arm2.json`)

## Checklist
- [x] No credentials, tokens or signed URLs
- [x] No private test examples or generations
- [x] No weight files (`*.safetensors`, `*.npz`, `*.pt`) and no `*.generations.jsonl`
- [x] Parent and evaluation ids match `delta-workspace.json`
- [x] `delta validate` passed on the submitted directory

## Results against the predictions
Arm 1's three predictions failed, in the same direction; arm 2's three held. Together the three points are a dose-response curve in α:

| α (effective temperature, effective presence penalty) | Public validation | Private test | Truncated of 39 | Output tokens |
|---|---|---|---|---|
| 1.000 (1.0, 1.5): the parent | 40.00 | 33.33 | 22 (56.4%) | 925,615 |
| 1.250 (0.8, 1.2): arm 2 | 23.33 | 22.22 | 29 (74.4%) | 1,056,680 |
| 1.667 (0.6, 0.9): arm 1 | 13.33 | 0.00 | 36 (92.3%) | 1,196,452 |
| published edit, for reference | 46.67 | 33.33 | 14 (35.9%) | 774,666 |

Arm 2 against its predictions: truncation 74.4% lies between the parent's 56.4% and arm 1's 92.3%; validation 23.33 is within 25 points of the parent; the private test is one item below the parent (2 of 9 against 3 of 9). Arm 1 against its predictions:

| Prediction | Pre-registered | Observed |
|---|---|---|
| 1. Truncation below the parent's 56.4% | yes | **92.3%** (36 of 39), against 56.4% for the parent and 35.9% for the published edit |
| 2. Public validation within 25 points of 40.00 and 46.67 | yes | **13.33**: −26.67 from the parent, −33.33 from the published edit |
| 3. Private test 33.33 ± one item | yes | **0.00** (0 of 9); the API reports `test_standard_error` 0.0 for it |

Reading. Both edits scale every logit by α, as the logit checks for both arms show. Under the vLLM v1 sampler order described above that is the parent at an effective temperature of 1/α with the presence penalty's effective size at 1.5/α; the server's order is not verified, and with the penalty applied after the temperature it would be temperature 1/α with the full penalty. Under the 32,768-token budget arm 1's setting ran the thinking trace into the cap on 36 of 39 generations, and output tokens rose from 925,615 to 1,196,452: the coupled change lengthened traces instead of shortening them (one run per arm). The two arms cannot say whether the lower effective penalty, the sharper distribution, or both did it, because they move together in every arm. What they establish is narrower: these two coupled settings (effective temperature 0.6 or 0.8 with the penalty scaled the same way) did not reproduce the published edit's `runtime-v5` result, which cut truncation to 35.9%. They do not rule out other decoding changes, including a sharper distribution with the penalty kept at 1.5, which a weight-only edit of this kind cannot express; the server would have to run such a cell (proposal 4 in `PROTOCOL.md`). Arm 2 sits between the parent and arm 1 on every column, so across these three points the effect of the coupled change is monotone in α.

Two facts for the protocol discussion. The server's `test_standard_error` is a sample estimate: it reports 0.0 when an entry scores 0 of 9, which understates the uncertainty of a 9-item test (the worst-case bound is 16.7 points). And on validation the parent and the published edit sit 26.67 and 33.33 points above arm 1, about 1.07 and 1.33 times the 25-point unpaired worst-case band at n = 30.

## Re-scored under `public-ladder500-8k-v2`
On 2026-11-04 the board moved to `public-ladder500-8k-v2` (18 public validation benchmarks, 1,110 items; a provisional private test of 500 items; 8,192 output tokens; the same sampler otherwise) and re-ran every version, these two arms included (results completed 23:32–23:48 UTC):

| Version | Public validation | Private test ± `test_standard_error` | `truncation_rate` | Output tokens |
|---|---|---|---|---|
| Original model | 41.83 | 18.6 ± 1.74 | 74.6% | 3,583,738 |
| Arm 2, α = 1.25 | 41.45 | 19.0 ± 1.76 | 75.6% | 3,567,886 |
| Arm 1, α = 1.667 | 41.89 | 17.2 ± 1.69 | 82.6% | 3,645,515 |
| Published edit | 41.26 | 19.0 ± 1.76 | 71.6% | 3,516,881 |

The `runtime-v5` validation and private-test drops do not reappear as gaps larger than the unpaired band: the four versions are within 0.63 points on validation and 1.8 points on the private test, where the band built from the reported standard errors is about 4.8 points (validation reports no standard error), and paired results from the evaluator would be needed to say whether any two differ. `truncation_rate` still rises with α (74.6%, 75.6%, 82.6%; one run per version). The `runtime-v5` results above rest on 30 validation and 9 private items and are not replicated here.
