# Dose-response of the published repetition-direction removal

## Hypothesis and scope
- Experiment directory: `research/abbyshekit-repetition-dose/`
- Model competition `qwen35-08b-v1`; parent version `a155dc2f-4075-48a3-9102-e1749f2b352f` (the published-edit reproduction); evaluation config `public-ladder500-8k-v2`, which the board moved to on 2026-11-04, re-scoring every version (results completed 23:32–23:48 UTC): 18 public validation benchmarks, a provisional private test of about 500 items (reported standard error 1.76 points), 8,192 output tokens, otherwise the same decoding. Under it the parent version scores 41.26 validation / 19.0 private with 71.6% of generations hitting the cap and 3,516,881 output tokens; the original model 41.83 / 18.6 / 74.6%; the two final-norm arms 41.45 / 19.0 and 41.89 / 17.2. The `runtime-v5` day-one figures (46.67 / 33.33 and 40.00 / 33.33 on a 9-item test) are superseded and are quoted below only where the published notes are compared with that board. Naming: the scripts and result files call that version `leader` and call the original model (`0ea62f40-7cd8-4d98-9ebb-484192fa05eb`, 40.00 / 33.33) `parent`; this text says "parent version" for the former and "original model" for the latter.
- Attribution: agent-assisted (Claude Code), human-reviewed. During the research the agent could see the public validation set and scores, the parent's local generations on public validation items at 1,024 and 4,096 output tokens, the published edit's work notes (below), and a locally built held-out set of fresh items used only for the local screen. It saw no private test material. The dose K was chosen by the local screen before any server run.

The published edit (Rikiya Takahashi, run `p50c`, [work notes](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes)) is `W = W0 − 6·u·(uᵀW0)` on 64 columns per layer of the layer-15 and layer-19 MLP down-projections, where `u` is a unit "repetition direction" found from matched productive and repetitive contexts. The notes record the dose curve on GSM8K and MATH train problems at the evaluation settings given in the run's [README](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md) (thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, at most 32,768 new tokens):

| Scale | Coverage | Paired gain (points) | Truncated | Whitespace-tail outputs |
|---|---|---|---|---|
| 1 | all columns | +12.5 (SE 2.6) | 91 → 74% | — |
| 2 | all columns | +24.6, +34.8 | 89 → 52% | 3 / 256 |
| 3 | all columns | +39.8 (SE 3.8) | → 36% (original not recorded) | 12 / 256 |
| 4 (16k cap) | all columns | +26.6 | 89 → 52% | 10 / 128 |
| 4 (16k cap) | top 64 | +26.6 | 90 → 48% | 2 / 128 |
| 6 | top 64 | +43.8, +51.6, +50.0 | 89 → 8–25% | 50 / 1,088 (4.6%) |

The run froze scale 6 under its own rule ("no further dose increase before independent confirmation"), not because a higher dose failed; the gain was still rising with scale and widening coverage at a fixed scale added degeneration without rescue. On this board under `runtime-v5` (32,768 tokens) the scale-6 edit truncated 14 of 39 generations against the original's 22; under `public-ladder500-8k-v2` (8,192 tokens) 71.6% of its generations hit the cap against the original's 74.6%, and the private-test scores are 19.0 against 18.6, a gap of 0.4 points against standard errors of 1.74–1.76. The server credits at least some answers that appear before the cut: under `runtime-v5` the final-norm × 1.667 arm scored 4 of 30 validation items while only 3 of its 39 generations finished. The CLI's scorer (`delta_cli/scoring.py`) takes the text after the last `</think>` as the answer, and the whole text when no `</think>` appears, which is consistent with that; the server's scorer was not inspected. One possible route to a gain at 8,192 tokens is therefore reaching the answer before the cut.

Hypothesis: one more step along the same direction on the same 64 neurons per layer, scale 6K with K > 1, cuts the cap-hit rate further on this board without changing what the model knows; the cost is whitespace-tail degeneration and damage on the non-math benchmarks, which the local screens and the per-benchmark validation scores measure. The edit is `leader + (K − 1)·(leader − parent)`: the sparse diff on the parent version touches a subset of the same 130,188 entries (127,701 at K = 1.25, 129,334 at K = 1.5 and 130,181 at K = 2.0; the rest round back to the parent version's bfloat16 value) and nothing else.

Arms: K = 1.5 (scale 9) and K = 2.0 (scale 12), chosen by the local screens below before any server run. K = 1.25 (scale 7.5) was built and screened too; at KL 0.0026 from the parent version it is within re-roll distance of it, so it was not submitted.

Pre-registered predictions, written before any server score, on `public-ladder500-8k-v2`. Arm 1, K = 1.5:
1. Cap-hit rate below the parent version's 71.6%.
2. Private test ≥ 19.0 (one standard error is 1.76 points; a difference under 3.5 points is inside noise and does not separate the arms).
3. Public validation ≥ 41.26 overall, and no more than 10 points below the parent version on ifeval, humaneval or mbpp (the degeneration indicators; 32–64 items each).
4. Output tokens below the parent version's 3,516,881.

Arm 2, K = 2.0, the high-variance probe at an untested scale:
1. Cap-hit rate below 71.6%.
2. Private test ≥ 17.2 (the lowest scored version on this config).
3. Public validation ≥ 40.0, and no more than 15 points below the parent version on ifeval, humaneval or mbpp; a larger drop marks scale 12 as past the peak.

What would falsify the hypothesis: arm 1's cap-hit rate at or above 71.6%, or its private test below 17.2. Sequential design: arm 2 is submitted after arm 1's result is in; if arm 1 trips its degeneration indicator, the second submission becomes K = 1.25 (built and screened, `results/manifest-k1.25.json`) instead of K = 2.0, and that switch is recorded here before it is submitted. Two arms on one curve give the board a dose-response pair at scales 6, 9 and 12.

## Reproduction
Parent `a155dc2f-4075-48a3-9102-e1749f2b352f`, config `public-ladder500-8k-v2`, server decoding temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 8,192 output tokens, thinking on (the local screens below were run under the earlier `runtime-v5` settings, which differ only in the 32,768-token cap); local generations use the CLI's per-sample seeds (1000 + sample).

From `delta competition init qwen35-08b-v1 --parent a155dc2f-4075-48a3-9102-e1749f2b352f --output experiment-leader` and `delta competition init qwen35-08b-v1 --output experiment` (the original model), copy this directory's scripts into `experiment-leader/` and work there:
```sh
cp /path/to/deltaleague/research/abbyshekit-repetition-dose/{dose_edit.py,dose_screen.py} experiment-leader/
cd experiment-leader
python dose_edit.py 1.5 edited-k1.5.safetensors ../experiment/model model    # arm 1, scale 9
python dose_edit.py 2.0 edited-k2.0.safetensors ../experiment/model model    # arm 2, scale 12
delta patch build --parent model --modified edited-k1.5.safetensors --output submission-k1.5
delta patch build --parent model --modified edited-k2.0.safetensors --output submission-k2.0
delta validate submission-k1.5 && delta validate submission-k2.0
```
Compute: one Apple M-series laptop, MPS, float32 for the screens. Expected: each manifest lists the two down-projection tensors (changed entries as above); edited checkpoint SHA-256 `baaf96f2a8616dbd…` (K = 1.5) and `5b6524ba8582c710…` (K = 2.0); artifact SHA-256 in `results/manifest-k1.5.json` (`107a7e31eb866335…`) and `results/manifest-k2.0.json` (`663962adbd158ab2…`); `manifest-k1.25.json` is the unsubmitted arm. The teacher-forced screen is `python dose_screen.py` run from the same directory; its header lists the local `delta evaluate` generation files it reads (traces of the original model and the parent version on public validation items at 1,024 and 4,096 tokens, plus two closed traces) and the commands that regenerate them. Those files are not in this PR, so the teacher-forced figures below are local estimates that a reader reproduces by regenerating the traces, not from the PR alone.

## Evidence
Local, before any server run (`results/dose-results.json`, `results/screen-*.json`):
The published edit is rank-1 on the 64 chosen columns per layer (top singular value carries 0.99999 of the energy) with scale 6 per column to bfloat16 precision (5.991–6.007; `dose_screen.py`, step 1); the two layers' directions have |cos| 0.849; the chosen columns are not the top columns by |uᵀW|, so coverage was not widened. On 16 local traces of the original model and the parent version (14 still thinking at their cap, 2 closed; `results/dose-results.json` holds the per-arm numbers, the script prints the trace counts), the layer-15 repetition component (the mean projection of the MLP output onto u over thinking positions, with u oriented so that the original model's mean is positive) moves monotonically with K: original +0.115, parent version −0.255, K = 1.25 −0.348, K = 1.5 −0.440, K = 2.0 −0.625; layer 19 is near zero from the parent version on (−0.037 to −0.054). Distance from the parent version, as KL per position on thinking and answer positions: K = 1.25 0.0026 / 0.0021, K = 1.5 0.0118 / 0.0089, K = 2.0 0.0506 / 0.0366; the original model is 0.027 / 0.018 from it, and a half-dose control (K = 0.5, in the same file) is 0.0078 / 0.0054 with its layer-15 component at −0.07. Top-1 agreement with the parent version: 0.986, 0.973, 0.941 on thinking positions against the original's 0.957. The `</think>` logit gap does not move with K (10th percentile 7.8–9.4 nats under the server's penalised top-20 across arms and task families), so the edit does not act by raising `</think>` directly; it acts by stopping repetition so a trace reaches its natural close.

Generation screen at 4,096 output tokens on six fresh held-out items, one sample each, server decoding settings (`results/screen-dose-*.json`): every arm truncates 5 of 6 at that cap (correct traces run about 9k tokens); the one item that closes finishes in 2,789 (parent version), 2,644 (K = 1.25), 2,132 (K = 1.5) and 2,018 (K = 2.0) tokens and is scored correct in every arm. Loop and degeneration counts are in `results/screen-dose-summary.json`: zero traces whose last 300 eight-grams are more than half repeats of earlier text, and zero whitespace tails, in any arm. The reports' own scores (33.33 for the parent version and K = 2.0, 16.67 for K = 1.25 and K = 1.5) are not used to rank the arms: at this cap they mostly record whether the right number appeared in a truncated GSM trace, which the server's scoring also credits, and with six items they cannot separate the arms.

The published notes give the cost side at scale 6 (temperature 0.6): 179 wrong-but-finished answers per 1,088 against 6 for the original, 53 no-answer finishes, 50 whitespace tails (4.6%). On this board the scale-6 edit's private-test gain over the original is 0.4 points at the 8,192-token cap (19.0 against 18.6, standard error 1.76) and its validation is 0.57 below, which this run cannot separate from zero; whether rescued traces become correct answers is the risk the arm-1 predictions bound.

Local estimates only; the generation screen is capped at 4,096 tokens, so it measures loop onset and degeneration, not closure at the server budget.

## Submission
- Arm 1 (K = 1.5): submission `a6ddce3e-b337-4249-b893-6dfd6a4c51dc`, submitted 2026-11-05T00:06Z; version `e75e2ed5-4b15-40d7-b7a9-75acb45faf14`, verified 2026-11-05T00:36Z on `public-ladder500-8k-v2`
- Arm 2 (K = 2.0): not submitted. Arm 1 met the falsification condition, so the second slot was not spent on a larger step along the same direction; the built patch and its manifest (`results/manifest-k2.0.json`) remain for anyone who wants that point.

## Results
Arm 1 against the parent version and the original model, all on `public-ladder500-8k-v2` (leaderboard and submission API, 2026-11-05):

| | Parent version (K = 1, scale 6) | Arm 1 (K = 1.5, scale 9) | Original model |
|---|---|---|---|
| Public validation | 41.26 | 39.59 | 41.83 |
| Private test ± `test_standard_error` | 19.0 ± 1.76 | **13.8 ± 1.54** | 18.6 ± 1.74 |
| `truncation_rate` | 71.6% | 72.2% | 74.6% |
| Output tokens | 3,516,881 | 3,547,814 | 3,583,738 |
| ifeval / humaneval / mbpp | 64.06 / 37.50 / 12.50 | 59.38 / 34.38 / 12.50 | 53.13 / 25.00 / 15.63 |
| bbh / gsm8k / math500 / mmlu_pro | 25.93 / 53.13 / 40.63 / 25.71 | 16.67 / 46.88 / 37.50 / 22.86 | 35.19 / 59.38 / 43.75 / 34.29 |

| Arm 1 prediction | Observed |
|---|---|
| 1. Cap-hit rate below 71.6% | failed: 72.2% |
| 2. Private test ≥ 19.0 | failed: 13.8 |
| 3. Validation ≥ 41.26; ifeval, humaneval and mbpp within 10 points of the parent version | validation failed (39.59); the degeneration indicator held (−4.69, −3.13, 0.00) |
| 4. Output tokens below 3,516,881 | failed: 3,547,814 |

The falsification condition is met: arm 1's private test (13.8) is below 17.2. Its drop of 5.2 points against the parent version is larger than 1.96·√(1.54² + 1.76²) = 4.58 points, the band built from the two marginal standard errors; that band assumes independence and is conservative when the two runs' per-item results on the shared items are non-negatively correlated, so under that assumption the drop is larger than the band; the paired interval was not available. (Prediction 2's parenthetical reads the marginal band as noise; it is an independence-based heuristic, and the paired difference the evaluator could report would be tighter.) The cap-hit rate did not fall: at 8,192 tokens the further step along the published direction moved the repetition component (local screen) but did not shorten generations on the server, and the losses are spread across reasoning benchmarks (bbh −9.26, gsm8k −6.25, mmlu_pro −2.86, math500 −3.13) rather than concentrated in the degeneration indicators. The same four benchmarks already fall from the original model to the parent version (bbh 35.19 → 25.93, gsm8k 59.38 → 53.13, math500 43.75 → 40.63, mmlu_pro 34.29 → 25.71) while ifeval and humaneval rise, so both steps along the direction score lower on these four benchmarks; those four benchmarks hold 54–70 items each, so each of these gaps is two to six items. What this establishes is narrow: on this config's private test, scale 9 on the published columns scores 5.2 points below scale 6, outside the independence band; validation is 1.67 points lower. It does not locate where between scale 6 and 9 the loss begins, and says nothing about doses below 6.

## Boundaries
- Only this experiment directory changed.
- No weights, credentials, private test material, signed URLs or raw generation logs.
- Local estimates and verified results are labelled as such above.
