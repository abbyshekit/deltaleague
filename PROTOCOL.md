# Evaluation protocol: proposals

Server-side proposals for the evaluator and the leaderboard, numbered so each can be taken or left alone. The CLI side of 4 and 5 (`delta grid`, `delta compare`, the research PR template) ships separately. Numbers are from the public API on 2026-11-04: `/api/v1/leaderboard?model=qwen35-08b-v1` and `/api/v1/evaluation-configs/runtime-v5`.

## Day one

| | Original | Published-edit reproduction |
|---|---|---|
| Validation (30 items: 10 each ARC-Challenge, GSM8K, MATH-500; 1 sample) | 40.00 | 46.67 |
| Private test | 33.33 | 33.33 |
| `test_standard_error` | 16.67 | 16.67 |
| Truncation rate (of 39 generations) | 56.4% (22) | 35.9% (14) |
| Output tokens | 925,615 | 774,666 |
| `estimated_gpu_cost_usd` (one RTX 5090) | 0.117 | 0.152 |

Both entries hold rank 1. Each run is 39 generations, 30 validation and 9 private test, and 16.67 = 50/√9. The validation gain is two more ARC answers out of ten (60 → 80); GSM8K and MATH-500 stay at 30. Generation settings: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 32,768 output tokens, thinking on. The 22 truncated generations at 32,768 tokens are 78% of the original's 925,615 output tokens.

## 1. Show the uncertainty on the board
`test_standard_error` is already in the API. Show it, and a validation standard error, next to each score, with the smallest difference two entries can show at 95% (about 2.8 × SE: 46 points on the private test today, 25 on validation). Entries inside that band are ties; the board already ranks them so, but the page does not say why. The evaluator holds per-item results for every entry, so differences between entries can also be paired on the same items, which is tighter than that bound.

## 2. Size the private test to the claims it has to settle
For a binary-scored item the worst-case standard error of a mean is 50/√n points:

| n | SE (points) | Resolves about |
|---|---|---|
| 9 | 16.7 | 46 |
| 100 | 5.0 | 14 |
| 400 | 2.5 | 7 |
| 1,600 | 1.25 | 3.5 |

At the measured $0.0030–0.0039 per generation, 400 items per benchmark across three benchmarks cost $3.60–4.70 per entry on one RTX 5090, about 31× today's 39 generations. Truncated generations dominate that cost, so a lower token cap on the test set moves it most. Several samples per item at temperature 1.0 cut the per-item variance for the same n; `samples` per example is already in the data format.

## 3. Rank on items the base model cannot have seen
The private test is drawn from public benchmark examples and is marked provisional. The ARC-Challenge, GSM8K and MATH test items have been public since 2018, 2021 and 2021; an edit can raise a score on them by sharpening recall rather than reasoning, and the board cannot tell the two apart. Proposal: build the ranking set from items written after the base model's data cutoff, or from templated variants of public items with recomputed answers; publish each entry's public-minus-private gap as the contamination indicator; seal the set with a hash published when the evaluation config is created; rotate it with each config version. The validation set stays public and stays reported, as the secondary number.

## 4. Score the original under a pre-registered decoding grid
The day-one edit's validation gain arrives with truncation falling from 56% to 36%: fewer generations hit the token cap, so more reach an answer. A decoding change (lower temperature, a lower presence penalty, thinking off) can do the same without touching a weight. Proposal: for each evaluation config, score the original model across a pre-registered decoding grid (the CLI's `DEFAULT_GRID`, six cells, hash recorded in the config) and show the best cell as an unranked reference row, "Original (decoding-tuned)". An entry that does not beat it has shown a decoding effect, not a weight-space gain. Cost: six evaluator runs per config, once.

## 5. Record the editor and show a scripted baseline
Entries declare human or agent (model, tool) in the research PR, and the board shows it. Per config, one unranked reference row for the best scripted edit: a fixed rule applied without search. Agent gains are then read against the simplest procedure, which is the comparison the competition is making.

## 6. Reference entries
Entries from the organiser, the evaluator maintainer or the author of the evaluation config appear as reference rows and are not ranked. The day-one reproduction belongs in that class.
