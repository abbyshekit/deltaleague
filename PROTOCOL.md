# Evaluation protocol: proposals

Server-side proposals for the evaluator and the leaderboard, numbered so each can be taken or left alone. The client side ships in separate PRs: `delta compare` (#1), `delta grid` (#2), the research PR template (#3) and `delta leaderboard --table` (#5). Numbers are from the public API on 2026-11-04 (`/api/v1/leaderboard?model=qwen35-08b-v1`, `/api/v1/evaluation-configs/runtime-v5`) and from deltaleague.ai/about, which says the private test is provisional and built from public benchmark examples.

## Day one

| | Original | Published-edit reproduction |
|---|---|---|
| Public validation (30 items: 10 each ARC-Challenge, GSM8K, MATH-500; 1 sample) | 40.00 | 46.67 |
| Private test | 33.33 | 33.33 |
| `test_standard_error` | 16.67 | 16.67 |
| Truncated generations (of 39) | 22 (56.4%) | 14 (35.9%) |
| Output tokens | 925,615 | 774,666 |
| `allocated_gpu_seconds` / `estimated_gpu_cost_usd` (one RTX 5090) | 735 / 0.117 | 960 / 0.152 |

Both entries hold rank 1. Each run is 39 generations, 30 public validation and 9 private test. The validation gain is two more ARC answers out of ten (60 → 80); GSM8K and MATH-500 stay at 30. The private-test standard error equals 50/√9 here; with 3 of 9 correct the sample standard deviation is exactly 0.5, so the API's formula cannot be told from this board. Cost follows allocated GPU seconds, not tokens: the edit emitted 16% fewer tokens and cost 30% more. Generation settings: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 32,768 output tokens, thinking on. If each truncated generation hit the cap, the 22 truncated ones are about 78% of the original's output tokens.

## 1. Show the uncertainty on the board
`test_standard_error` is already in the API. Show it, and a public-validation standard error, next to each score, with the smallest difference two entries can show at 95% (1.96·√2·SE: 46 points on the private test today, 25 on public validation at the worst case of 50/√30). Entries inside that band tie. Both day-one entries hold rank 1 already; the page should say why. The evaluator holds per-item results for every entry, so differences between entries can also be paired on the same items, which is tighter than that bound. Client side: `delta leaderboard --table` (#5).

## 2. Size the private test to the claims it has to settle
For a binary-scored item the worst-case standard error of a mean is 50/√n points:

| n | SE (points) | Resolves about (1.96·√2·SE) |
|---|---|---|
| 9 | 16.7 | 46 |
| 100 | 5.0 | 14 |
| 400 | 2.5 | 7 |
| 1,600 | 1.25 | 3.5 |

Cost at larger n is not measured. A linear extrapolation of today's $0.0030–0.0039 per generation gives $3.60–4.70 for 400 items per benchmark across the three benchmarks on one RTX 5090, about 31× today's 39 generations. The day-one figures include a fixed allocation overhead, so the per-generation cost at scale is lower and should be measured once. Several samples per item at temperature 1.0 cut the per-item variance for the same n; `samples` per example is already in the data format.

## 3. Rank on items the base model cannot have seen
The private test is built from public benchmark examples and is marked provisional. The ARC-Challenge (2018), GSM8K (2021) and MATH (2021; MATH-500 is its 2023 subset) test items have been public for years. An edit can raise a score on them by sharpening recall rather than reasoning. The board cannot tell the two apart.

Proposal: build the private test from items written after the base model's data cutoff, or from templated variants of public items with recomputed answers. Publish each entry's public-minus-private gap as the contamination indicator. Seal the set with a hash published when the evaluation config is created. Rotate it with each config version. Public validation stays public and stays reported, as the secondary number.

## 4. Score the original under a pre-registered decoding grid
The day-one edit's validation gain is two ARC items, and its truncation drop is eight generations; both sit inside the tie band of proposal 1. The hypothesis a larger test would check is that fewer generations hitting the token cap is what moves the score, and a decoding change (lower temperature, a lower presence penalty, thinking off) can do that without touching a weight. Proposal: for each evaluation config, score the original model across a pre-registered decoding grid (the six cells of `DEFAULT_GRID` in #2, hash recorded in the config) and show the best cell as an unranked reference row, "Original (decoding-tuned)". An entry that does not beat it has shown a decoding effect, not a weight-space gain. Cost: six evaluator runs per config, once.

## 5. Record the editor and show a scripted baseline
Entries declare human or agent (model, tool) in the research PR (#3), and the board shows it. Per config, one unranked reference row for the best scripted edit: a fixed rule applied without search. Agent gains are then read against the simplest procedure.

## 6. Reference entries
Entries from the organiser, the evaluator maintainer or the author of the evaluation config appear as reference rows and are not ranked, because the private test and the evaluator are visible to them. The day-one reproduction belongs in that class.
