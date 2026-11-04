"""Exact check: with the final norm gain scaled by ALPHA, next-token logits scale by ALPHA (tied embeddings, no bias, no soft-capping)."""
import sys
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
alpha = float(sys.argv[1])
tok = AutoTokenizer.from_pretrained('model', local_files_only=True)
config = AutoConfig.from_pretrained('model', local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained('model', config=config, local_files_only=True, dtype=torch.float32).eval()
norm = model.model.language_model.norm
prompts = ['What is 17 * 23?', 'Convert the point (0,3) in rectangular coordinates to polar coordinates.', 'Which is the most likely effect of faster planetary rotation?']
with torch.inference_mode():
    rows = []
    for p in prompts:
        text = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok(text, return_tensors='pt', add_special_tokens=False)
        w0 = norm.weight.clone()
        base = model(**ids).logits[0, -1].clone()
        norm.weight.copy_((alpha * (1 + w0) - 1).to(torch.bfloat16).float())
        edited = model(**ids).logits[0, -1].clone()
        norm.weight.copy_(w0)
        ratio = edited / base
        rows.append((p[:40], float((edited - alpha * base).abs().max()), float(base.abs().max()), float(ratio.median()), float(ratio.min()), float(ratio.max()), int(torch.argmax(base) == torch.argmax(edited))))
for r in rows:
    print(f'{r[0]:<42} max|edited-α·base|={r[1]:.4f} (max|base|={r[2]:.2f}); ratio median {r[3]:.4f} min {r[4]:.4f} max {r[5]:.4f}; argmax same={r[6]}')
