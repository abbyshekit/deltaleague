# Held-out set v1: pipeline and seal (competition proposal)

The pipeline that built a held-out test set for proposal 3 of `PROTOCOL.md`, with the sealed hash of the result. **The items themselves are not in this repository**: a held-out set that is public is not held out; hidden evaluation material belongs in the private platform repository (CONTRIBUTING.md) with the platform's data on R2. The set (`heldout-v1.json`, 313 items in the `validation.json` format, SHA-256 in `heldout-v1.sha256`) and its private sidecar (solutions, programs, solver answers, option rotations) are delivered to the maintainer out of band; `report.json` carries the counts.

Exposure note: an earlier revision of this branch carried the items for about an hour on a public fork before this rewrite. Treat v1 as exposed for ranking purposes: run the pipeline again for a fresh set before using one on the private test, and keep v1 as a development set. Three families mirror the public validation set's format and scorers: `arc_fresh` (science multiple choice, scorer `choice`), `gsm_fresh` (grade-school word problems, scorer `math`), `math_fresh` (competition problems, scorer `math`). Every prompt ends with the platform's own instruction line. A delivered file runs locally with `delta evaluate --validation heldout-v1.json` (#10), which records its hash in the report.

## How it was built
1. Writers (Claude Opus) wrote ten original items per batch for an assigned topic, with a worked solution and, for `gsm_fresh`, a Python `solve()` that computes the answer from the stated quantities.
2. Two solvers (Claude Sonnet) solved each item blind; the second was told to hunt for ambiguity, a second defensible answer, or a missing quantity.
3. A critic (Claude Opus) rejected items that are a near-duplicate of a specific item it could name from ARC, GSM8K, MATH, MMLU, GPQA, AQuA, SVAMP, ASDiv, MathQA, AIME/AMC or a well-known textbook, items with more than one defensible answer, and items guessable from surface cues. In the first pass the critic also rejected items for sharing a *template* with public items (seasons, frost wedging, ratio changes); that rule was dropped for the second pass, because the public benchmarks are built from the same templates and a held-out set must exclude memorized items, not topics.
4. An item was accepted only when the writer and both solvers agreed on the answer (a Sonnet judge decided equivalence when two LaTeX forms differed) and the critic kept it. v1 was accepted under the earlier gate, which vetoed on solver B's ambiguity flag and solver A's lack of confidence only. The scripts in this directory apply the stricter gate: an item is rejected when either solver reports `ambiguous: true` or does not report `confident: true`, and each solver's `confident`, `ambiguous` and `issue` values are kept as `solver_flags` in the item's private provenance and in its rejection record.
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
### Generate a fresh set
Prerequisites: Node 18+, the Claude Code CLI (`claude`) logged in, Python 3.12 with `math-verify`, and network access for `fetch_public.py`. `run_workflow.mjs` has no npm dependencies.

```
python fetch_public.py
node run_workflow.mjs workflow-pass2.js families-pass1.json --out /private/path/pass1.json
python build_heldout.py --exclude <earlier set, e.g. heldout-v1.json> /private/path/pass1.json /private/path/out
```
- `workflow-pass1.js` and `workflow-pass2.js` are written for the Claude Code workflow runtime (globals `agent`, `parallel`, `log`, `args`). `run_workflow.mjs <script> <families.json> --out <result.json>` supplies those globals outside Claude Code: each `agent()` call runs `claude -p` with the schema as `--json-schema` and the prompt on stdin, retries once, and resolves to `null` on a second failure. Options: `--concurrency N` (default 4, applies to agent processes), `--timeout-min M` (default 20; a call that runs longer is killed and counts as a failed attempt), `--model-override MODEL` (replaces every agent's model, for cheap smoke tests), `--dry-run` (offline stub agents, no model calls). Agents run in a temporary directory, so no repository's `CLAUDE.md` is loaded; user-level Claude Code settings still apply. A one-family smoke run (`math_fresh`, target 1, `--model-override haiku`) made 4 calls and cost $0.02.
- `workflow-pass2.js` holds the current critic rule; use it for new sets. `workflow-pass1.js` keeps the first pass's critic rule (with the template veto) as the record of how that pass ran; both scripts apply the stricter solver gate.
- `families-pass1.json` is the configuration of v1's first pass: per family the topics, spec, answer and program rules, critic rule, public-benchmark examples, `target` and `batchesPerRound`. `families-pass2.json` is the top-up run (`arc_fresh` 85, `math_fresh` 20, with fresh `arc_fresh` topics). They contain no generated items. For a new set, replace the `topics` lists and set the targets. A family stops at its target, after 8 rounds, or after two consecutive rounds with fewer than two new accepted items.
- `--exclude` (repeatable) takes an earlier set in `validation.json` format. Its prompts join the 10-word overlap veto and the duplicate check, so a fresh set cannot repeat an exposed one; the counts are recorded in `report.json`.
- `build_heldout.py` always names its outputs `heldout-v1.json`, `heldout-v1.private.json` and `heldout-v1.sha256`, whatever the set; write a fresh set to its own `<out_dir>` and rename it on delivery. The result JSONs, these outputs and the veto and rejection records contain item text. Keep them outside the repository. `fetch_public.py` writes the public splits to `public/` beside the script; do not commit them.

### Rebuild v1 from its workflow results
```
python fetch_public.py
python build_heldout.py <workflow results...> <out_dir>
```
`fetch_public.py` downloads the three public test splits into `public/` and prints each row count and SHA-256. `build_heldout.py` takes the result JSONs the two workflow scripts (`workflow-pass1.js`, `workflow-pass2.js`) produce, applies the deterministic vetoes and writes the sealed set, the private file, the veto and rejection records, `heldout-v1.sha256` and `report.json` into `<out_dir>`.

The committed set was built from runs `wf_616c5f47-fbe` (pass 1) and `wf_7a2c8833-e67` (pass 2). Their result JSONs, and the per-item rejection log, are delivered with the set rather than committed, because they contain item text.

The build executes model-written Python (the `gsm_fresh` `solve()` programs) only
inside the same pinned CPython/WASI sandbox used for public coding benchmarks.
The AST screen remains a format policy, not the isolation boundary. WASI exposes
only its read-only standard library, with no host files, host environment or
network, and bounds memory, fuel, elapsed time and output. The numeric answer
check runs inside that sandbox. Missing dependencies, a missing runtime or a
failed preflight stop the build; there is no host-Python fallback.

Install the current repository package and pinned Wasmtime in a separate Python
3.12 environment before running the build. Run these setup commands from the
repository root:

```sh
python3.12 -m venv .venv-heldout
.venv-heldout/bin/pip install . math-verify==0.9.0 wasmtime==50.0.1 pytest
.venv-heldout/bin/python -m delta_cli.public_eval.install_wasi "$PWD/.wasi-heldout"
export DELTA_WASI_ROOT="$PWD/.wasi-heldout"
```

Use `.venv-heldout/bin/python` for `build_heldout.py`. The runtime installer checks
the archive SHA-256. Program errors and wrong answers veto individual items;
sandbox infrastructure failures abort the build. The ordinary tests cover that
fail-closed behavior. With the runtime installed, also run:

```sh
DELTA_TEST_HELDOUT_WASI=1 .venv-heldout/bin/python -m pytest -q tests/test_heldout_programs.py
```
