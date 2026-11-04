"""Scale the language model's final RMSNorm gain by ALPHA. Qwen3.5's RMSNorm is zero-centred (gain = 1 + w), so the stored weight becomes ALPHA * (1 + w) - 1; under fixed decoding this equals temperature 1/ALPHA with the presence penalty also divided by ALPHA."""
import os
import sys
import torch
from safetensors.torch import load_file, save_file
alpha, out = float(sys.argv[1]), sys.argv[2]
if not alpha > 0:
    sys.exit(f'alpha must be > 0, got {alpha}')
if os.path.exists(out):
    sys.exit(f'refusing to overwrite existing output file: {out}')
name = 'model.language_model.norm.weight'
weights = load_file('model/model.safetensors')
before = weights[name].clone()
weights[name] = (alpha * (1 + before.float()) - 1).to(torch.bfloat16)
save_file(weights, out)
gain = (1 + weights[name].float()) / (1 + before.float())
print(f'{name}: {int((weights[name] != before).sum())} of {before.numel()} entries changed; effective gain min {gain.min():.4f} max {gain.max():.4f} (target {alpha:.4f})')
