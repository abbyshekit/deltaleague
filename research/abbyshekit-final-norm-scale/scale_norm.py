"""Scale the language model's final RMSNorm by ALPHA: under fixed decoding this equals temperature 1/ALPHA with the presence penalty also divided by ALPHA."""
import sys
import torch
from safetensors.torch import load_file, save_file
alpha, out = float(sys.argv[1]), sys.argv[2]
name = 'model.language_model.norm.weight'
weights = load_file('model/model.safetensors')
before = weights[name].clone()
weights[name] = (weights[name].float() * alpha).to(torch.bfloat16)
save_file(weights, out)
changed = int((weights[name] != before).sum())
print(f'{name}: {changed} of {before.numel()} entries changed; mean |w| {before.float().abs().mean():.4f} -> {weights[name].float().abs().mean():.4f}')
