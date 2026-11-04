"""Scale the published repetition-direction removal.

Usage: dose_edit.py K OUT.safetensors [PARENT_DIR] [LEADER_DIR]
The published edit (Rikiya Takahashi's run p50c) is W = W0 - 6 u (u^T W0) on 64 columns per layer of
model.language_model.layers.{15,19}.mlp.down_proj.weight, where u is a unit repetition direction. Delta = leader - parent
holds exactly that change; the output is leader + (K - 1) * Delta in float32 with one bfloat16 cast, so the scale 6 becomes 6K.
K = 1 reproduces the leader. Every other tensor is untouched. PARENT_DIR holds the original model (default ../experiment/model),
LEADER_DIR the published-edit version (default model); the script checks that Delta has the published form (scale 6 on the
changed columns) and stops if the two directories look swapped or the edit is not the published one.
"""
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

if len(sys.argv) < 3:
    sys.exit(__doc__)
k, out = float(sys.argv[1]), sys.argv[2]
parent = sys.argv[3] if len(sys.argv) > 3 else '../experiment/model'
leader = sys.argv[4] if len(sys.argv) > 4 else 'model'
names = [f'model.language_model.layers.{layer}.mlp.down_proj.weight' for layer in (15, 19)]
weights = load_file(f'{leader}/model.safetensors')
with safe_open(f'{parent}/model.safetensors', 'pt') as f:
    base = {name: f.get_tensor(name) for name in names}
for name in names:
    w, w0 = weights[name].float(), base[name].float()
    delta = w - w0
    columns = (delta != 0).any(0).nonzero().flatten()
    u = torch.linalg.svd(delta[:, columns], full_matrices=False)[0][:, 0]
    scale = (-(u @ delta[:, columns]) / (u @ w0[:, columns])).median().item()
    if not 5.5 < scale < 6.5:
        sys.exit(f'{name}: the edit between the two directories has scale {scale:.3f} along its direction, not 6; '
                 'PARENT_DIR and LEADER_DIR may be swapped, or the edit is not the published one')
    new = (w + (k - 1) * delta).to(torch.bfloat16)
    print(f'{name}: {len(columns)} columns, {int((delta != 0).sum())} entries changed by the published edit (scale {scale:.3f}); '
          f'relative change vs the original model {(new.float() - w0).norm() / w0.norm():.4f} (published edit {delta.norm() / w0.norm():.4f})')
    weights[name] = new
save_file(weights, out, metadata={'format': 'pt'})
