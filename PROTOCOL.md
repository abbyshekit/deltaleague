# Evaluation protocol: proposals

Server-side proposals for the evaluator and the leaderboard, numbered so each can be taken or left alone. The client side ships in separate PRs: `delta compare` (#1), `delta grid` (#2) and `delta leaderboard --table` (#5); the research template is `.github/PULL_REQUEST_TEMPLATE/research.md`. Numbers are from the public API (`/api/v1/leaderboard?model=qwen35-08b-v1`, `/api/v1/evaluation-configs/<id>`) on 2026-11-04 and 2026-11-05, and from deltaleague.ai/about, which says the private test is provisional and built from public benchmark examples.

## Day one (`runtime-v5`)

| | Original | Published-edit reproduction |
|---|---|---|
| Public validation (30 items: 10 each ARC-Challenge, GSM8K, MATH-500; 1 sample) | 40.00 | 46.67 |
| Private test | 33.33 | 33.33 |
| `test_standard_error` | 16.67 | 16.67 |
| Truncated generations (of 39) | 22 (56.4%) | 14 (35.9%) |
| Output tokens | 925,615 | 774,666 |
| `allocated_gpu_seconds` / `estimated_gpu_cost_usd` (one RTX 5090) | 735 / 0.117 | 960 / 0.152 |

Both entries held rank 1. Each run is 39 generations, 30 public validation and 9 private test. The validation gain is two more ARC answers out of ten (60 → 80); GSM8K and MATH-500 stay at 30. The private-test standard error equals 50/√9 here; with 3 of 9 correct the sample standard deviation is exactly 0.5, so the API's formula cannot be told from these two rows. A later entry that scored 0 of 9 was reported with `test_standard_error` 0.0, so the API uses a sample estimate, which reads as certainty at the floor; the worst-case 50/√n is the honest figure for a 9-item test. Cost follows allocated GPU seconds, not tokens: the edit emitted 16% fewer tokens and cost 30% more. Generation settings: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 32,768 output tokens, thinking on. If each truncated generation hit the cap, the 22 truncated ones are about 78% of the original's output tokens.

## Re-scored (`public-ladder500-8k-v2`)
On 2026-11-04 the board moved to `public-ladder500-8k-v2` and re-ran every version (results completed 23:32–23:48 UTC): 18 public validation benchmarks (1,110 items), a provisional private test whose reported standard errors (1.54–1.76) match 500 binary items, 8,192 output tokens, the same sampler otherwise.

| | Original | Published-edit reproduction |
|---|---|---|
| Public validation | 41.83 | 41.26 |
| Private test ± `test_standard_error` | 18.6 ± 1.74 | 19.0 ± 1.76 |
| `truncation_rate` | 74.6% | 71.6% |
| Output tokens | 3,583,738 | 3,516,881 |
| `allocated_gpu_seconds` / `estimated_gpu_cost_usd` | 1,391 / 0.233 | 1,277 / 0.214 |

The day-one validation gain of 6.67 points does not reappear: the two versions are 0.57 apart on validation and 0.4 apart on the private test. Proposals 1 and 2 below were written against `runtime-v5`; this config moves the private test from 9 items to 500, the direction proposal 2 asks for.

## 1. Show the uncertainty on the board
`test_standard_error` is already in the API. Show it, and a public-validation standard error, next to each score. Marginal standard errors alone cannot say whether two entries differ: both are scored on the same items, so the variance of their difference includes the covariance between them. The band 1.96·√(se_a² + se_b²) assumes independence (on `runtime-v5` it was 46 points on the private test and 25 on public validation at the worst case of 50/√30); a gap inside it is not evidence that two entries are equal. The evaluator holds per-item results for every entry, so it can publish the paired difference and its interval against the parent version and against the top entry (a paired bootstrap, or McNemar's test for binary items), which is tighter than a bound built from marginal errors whenever the two entries' per-item results are positively correlated, as results on shared items usually are. Client side: `delta leaderboard --table` (#5) shows the marginal errors and labels its comparison as an independence-based heuristic.

## 2. Size the private test to the claims it has to settle
For a binary-scored item the worst-case standard error of a mean is 50/√n points. The last column is the unpaired worst case; a paired comparison on shared items is tighter when per-item results are positively correlated.

| n per benchmark | SE (points) | Unpaired band (1.96·√2·SE) |
|---|---|---|
| 9 | 16.7 | 46 |
| 100 | 5.0 | 14 |
| 400 | 2.5 | 7 |
| 1,600 | 1.25 | 3.5 |

Cost at larger n is now measured: on `public-ladder500-8k-v2` an entry's run (1,110 public validation items and 500 private items at 8,192 tokens) cost $0.21–0.30 and 1,080–1,709 generation seconds on one RTX 5090, against $0.12–0.15 for the 39 generations of `runtime-v5`. Several samples per item at temperature 1.0 cut the per-item variance for the same n; `samples` per example is already in the data format.

## 3. Rank on items the base model cannot have seen
The private test is built from public benchmark examples and is marked provisional. The ARC-Challenge (2018), GSM8K (2021) and MATH (2021; MATH-500 is its 2023 subset) test items have been public for years. An edit can raise a score on them by sharpening recall rather than reasoning. The board cannot tell the two apart.

Proposal: build the private test from items written after the base model's data cutoff, or from templated variants of public items with recomputed answers. Publish each entry's public-minus-private gap as the contamination indicator. Seal the set with a hash published when the evaluation config is created. Rotate it with each config version. Public validation stays public and stays reported, as the secondary number.

## 4. Score the original under a pre-registered decoding grid
The day-one edit's validation gain is two ARC items (6.67 points on 30 items, against an unpaired band of 25) and its truncation drop is eight of 39 generations (20.5 points, against 22); both are inside the unpaired bands of proposal 1. The hypothesis a larger test would check is that fewer generations hitting the token cap is what moves the score, and a decoding change (lower temperature, a lower presence penalty, thinking off) can do that without touching a weight. Proposal: for each evaluation config, score the original model across a pre-registered decoding grid (the six cells of `DEFAULT_GRID` in #2, with each cell's effective generation settings, not only its overrides, recorded in the config) and show the best cell as an unranked reference row, "Original (decoding-tuned)". An entry that does not beat it has not demonstrated an advantage over the decoding-tuned reference under this evaluation; the comparison does not by itself say what mechanism an edit uses. Cost: six evaluator runs per config, once.

## 5. Record the editor and show a scripted baseline
Entries declare human or agent (model, tool) in the research PR's attribution field, and the board shows it. Per config, one unranked reference row for the best scripted edit: a fixed rule applied without search. Agent gains are then read against the simplest procedure.

## 6. Reference entries
Entries from the organiser, the evaluator maintainer or the author of the evaluation config appear as reference rows and are not ranked, because the private test and the evaluator are visible to them. The day-one reproduction belongs in that class.

## 7. More base models
One competition, one 0.8B checkpoint, cannot separate an edit that improves a model from an edit that fits one checkpoint. Proposal: add a second family and a second size, chosen by one rule: a permissive licence and no gate, so patches can be published in full. Licence and gating as reported by the Hugging Face API on 2026-11-04:

| Candidate | Licence | Gated | Why |
|---|---|---|---|
| `Qwen/Qwen3.5-2B` | Apache-2.0 | no | same family, larger: does an edit transfer across size? |
| `allenai/OLMo-2-0425-1B` | Apache-2.0 | no | training data is public (Dolma), so proposal 3's contamination check can be run against the corpus rather than inferred |
| `HuggingFaceTB/SmolLM3-3B` | Apache-2.0 | no | second family with a published training recipe |
| `microsoft/Phi-4-mini-instruct` | MIT | no | third family, instruct-tuned (a `-reasoning` sibling exists under the same licence) |
| `ibm-granite/granite-4.0-h-micro` | Apache-2.0 | no | attention and Mamba blocks in one stack (`layer_types` in its config): does the edit recipe depend on the block type? |

Excluded under the rule: `google/gemma-3-1b-it` (Gemma licence, gated) and `meta-llama/Llama-3.2-1B-Instruct` (Llama 3.2 licence, gated). The cheapest first step is OLMo-2-1B: it is the one candidate with a fully public corpus and the tooling to search it for the test items.
