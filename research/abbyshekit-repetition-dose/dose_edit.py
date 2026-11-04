"""Scale the published repetition-direction removal.

Usage: dose_edit.py K OUT.safetensors [PARENT_DIR] [LEADER_DIR]
The published edit (Rikiya Takahashi's run p50c) is W = W0 - 6 u (u^T W0) on 64 columns per layer of
model.language_model.layers.{15,19}.mlp.down_proj.weight, where u is a unit repetition direction. Delta = leader - parent
holds exactly that change; the output is leader + (K - 1) * Delta in float32 with one bfloat16 cast, so the scale 6 becomes 6K.
K = 1 reproduces the leader. Every other tensor is untouched.
"""
import sys

import torch
from safetensors.torch import load_file, save_file

k, out = float(sys.argv[1]), sys.argv[2]
parent = sys.argv[3] if len(sys.argv) > 3 else '../experiment/model'
leader = sys.argv[4] if len(sys.argv) > 4 else 'model'
names = [f'model.language_model.layers.{layer}.mlp.down_proj.weight' for layer in (15, 19)]
weights = load_file(f'{leader}/model.safetensors')
base = load_file(f'{parent}/model.safetensors')
for name in names:
    w, w0 = weights[name].float(), base[name].float()
    delta = w - w0
    columns = int((delta != 0).any(0).sum())
    new = (w + (k - 1) * delta).to(torch.bfloat16)
    print(f'{name}: {columns} columns, {int((delta != 0).sum())} entries changed by the published edit; '
          f'relative change vs parent {(new.float() - w0).norm() / w0.norm():.4f} (leader {delta.norm() / w0.norm():.4f})')
    weights[name] = new
save_file(weights, out, metadata={'format': 'pt'})
