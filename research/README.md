# Research directory

One directory per research PR: `research/<github-user>-<slug>/`. Open the PR with `gh pr create --repo ilblackdragon/deltaleague --template research.md`, or copy `.github/PULL_REQUEST_TEMPLATE/research.md` into the PR body.

Contents:
- `README.md`: the filled template.
- The edit or reproduction code, runnable from a fresh `delta competition init` workspace.
- `results/`: the `delta evaluate`, `delta compare` and `delta grid` JSON reports. The reports carry scores, hashes and settings; `*.generations.jsonl` files stay out.

Never commit weights, private test data, credentials or signed URLs. `.gitignore` excludes `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.npz`, `*.generations.jsonl`, `.delta/` and `auth.json`.

The CLI records the PR link with the submission and does not read the PR.
