"""Teacher-forced screen of dose arms of the published repetition-direction edit (forward passes only; no generation).

Run from a `delta competition init --parent a155dc2f-4075-48a3-9102-e1749f2b352f` workspace (`model/` = the parent version,
called `leader` here) that sits next to a plain `delta competition init` workspace at ../experiment (`model/` = the original
model, called `parent` here). Usage: python dose_screen.py [step1|full]. Step 1 decomposes leader − parent (rank, scale,
column ranks); step 2 writes edited-dose-k{0.5,1.25,1.5,2.0}.safetensors; step 3 replays local generation files through
each arm and records the repetition component at layers 15/19, the penalised </think> gap, the closure hazard, and KL and
top-1 agreement against the leader. Inputs read from ../experiment, all written by `delta evaluate` on public validation
items (never private test material); regenerate with the commands shown, from ../experiment unless noted:
  ctrl-blog.generations.jsonl              delta evaluate --weights <published edit> --limit 2 --max-new-tokens 1024
  diag-4k.generations.jsonl                delta evaluate --limit 1 --max-new-tokens 4096
  screen-close-1.3.generations.jsonl       a 1,024-token run of an unrelated local arm; any extra traces serve the same role
  ctrl-grid/temp-0.6-pp-0.9.generations.jsonl   delta grid --grid grid-alpha.json (closed traces for the answer-phase gate)
  heldout-v1.json                          prompts for any trace generated on local held-out items (none were used here)
Traces whose retokenised length differs from the recorded token count by more than 2 are skipped and reported.
Sign convention: the SVD fixes u only up to sign, so after the screen u (and every reported loop component) is oriented so
that the original model's mean component over thinking positions is positive; the edit then reads as moving it negative.
"""
import json, time, math, sys, gc
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
T0 = time.time(); EX = '../experiment/'; THINK = 248069; PP = 1.5; LAY = (15, 19); MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
assert MODE in ('step1', 'full'), 'usage: dose_screen.py [step1|full]'
KEY = lambda l: f'model.language_model.layers.{l}.mlp.down_proj.weight'
LP, PP_ = 'model/model.safetensors', EX + 'model/model.safetensors'
def get(path, k):
    with safe_open(path, 'pt') as f: return f.get_tensor(k)
# ---- step 1
S1 = {}; Wl, Wp, U, SMED, ORDER, CHOSEN = {}, {}, {}, {}, {}, {}
for l in LAY:
    wl, wp = get(LP, KEY(l)), get(PP_, KEY(l)); Wl[l], Wp[l] = wl, wp
    D = wl.float() - wp.float(); cols = (D != 0).any(0).nonzero().flatten(); CHOSEN[l] = cols
    sv = torch.linalg.svd(D[:, cols], full_matrices=False); e = (sv[1] ** 2 / (sv[1] ** 2).sum())[:3].tolist()
    u = sv[0][:, 0]; Dc, Wc = D[:, cols], wp.float()[:, cols]
    s = -(u @ Dc) / (u @ Wc); U[l] = u; SMED[l] = s.median().item()
    proj = (u @ wp.float()).abs(); order = proj.argsort(descending=True); ORDER[l] = order
    rank = torch.empty_like(order); rank[order] = torch.arange(len(order)); rk = (rank[cols] + 1)
    S1[l] = dict(n_cols=len(cols), cols=cols.tolist(), n_entries=int((D != 0).sum()), sv_energy_top3=e, s_median=SMED[l], s_min=s.min().item(), s_max=s.max().item(),
                 rank_min=int(rk.min()), rank_max=int(rk.max()), rank_median=rk.float().median().item(), in_top64=int((rk <= 64).sum()), ranks_sorted=sorted(rk.tolist()))
S1['cos_u15_u19'] = abs(float(U[15] @ U[19]))
# all other tensors identical?
with safe_open(LP, 'pt') as a, safe_open(PP_, 'pt') as b:
    S1['other_tensors_differ'] = [k for k in a.keys() if k not in (KEY(15), KEY(19)) and not torch.equal(a.get_tensor(k), b.get_tensor(k))]
for l in LAY: print(l, {k: v for k, v in S1[l].items() if k not in ('cols', 'ranks_sorted')}); print('  cols', S1[l]['cols']); print('  ranks', S1[l]['ranks_sorted'])
print('cos', S1['cos_u15_u19'], 'other differing tensors', S1['other_tensors_differ'])
if MODE == 'step1': sys.exit()
# ---- step 2: arms
top64 = all(S1[l]['in_top64'] >= 60 for l in LAY)
ARMS = {'parent': {l: Wp[l] for l in LAY}, 'leader': {l: Wl[l] for l in LAY}}
for k in (0.5, 1.25, 1.5, 2.0):
    ARMS[f'k{k}'] = {l: (Wl[l].float() + (k - 1) * (Wl[l].float() - Wp[l].float())).to(torch.bfloat16) for l in LAY}
if top64:
    for name, n in (('cov128', 64), ('cov256', 192)):
        ARMS[name] = {}
        for l in LAY:
            C = set(CHOSEN[l].tolist()); extra = [c for c in ORDER[l].tolist() if c not in C][:n]; W = Wl[l].float(); ix = torch.tensor(extra)
            W[:, ix] -= SMED[l] * U[l][:, None] * (U[l] @ W[:, ix])[None]; ARMS[name][l] = W.to(torch.bfloat16)
print('top64', top64, 'arms', list(ARMS))
base = load_file(LP)
for name, ws in ARMS.items():
    if name in ('parent', 'leader'): continue
    sd = dict(base)
    for l in LAY: sd[KEY(l)] = ws[l]
    save_file(sd, f'edited-dose-{name}.safetensors', metadata={'format': 'pt'})  # always rewritten so the file matches what step 3 screens
    del sd
del base; gc.collect()
# ---- step 3
from transformers import AutoModelForImageTextToText, AutoTokenizer
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('model', local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained('model', local_files_only=True, dtype=torch.float32).eval().to(dev)
lm = model.model.language_model; E = lm.embed_tokens.weight.detach()
val = json.load(open('validation.json')); prompts = {e['id']: e['prompt'] for v in val.values() for e in v}
for v in json.load(open(EX + 'heldout-v1.json')).values(): prompts.update({e['id']: e['prompt'] for e in v})
cap = {}; lm.norm.register_forward_hook(lambda m, i, o: cap.__setitem__('h', o.detach()))
for l in LAY: lm.layers[l].mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o.detach()))
TR = []
for f, mode in (('ctrl-blog', 'think'), ('diag-4k', 'think'), ('screen-close-1.3', 'any'), ('ctrl-grid/temp-0.6-pp-0.9', 'closed')):
    for line in open(EX + f + '.generations.jsonl'):
        r = json.loads(line); has = '</think>' in r['text']
        if (not has and mode == 'closed') or (has and mode == 'think'): continue
        text = tok.apply_chat_template([{'role': 'user', 'content': prompts[r['example_id']]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        p = tok(text, add_special_tokens=False)['input_ids']; g = tok(r['text'], add_special_tokens=False)['input_ids']
        if abs(len(g) - r['tokens']) > 2: print('skipped (retokenised length mismatch)', f, r['example_id'], len(g), r['tokens']); continue
        gt = torch.tensor(g, device=dev); k = (gt == THINK).nonzero()[0].item() if has else None
        TR.append(dict(name=f'{f}:{r["example_id"]}', fam=r['example_id'].split('_')[0].split('-')[0], ids=torch.tensor([p + g], device=dev), P=len(p), g=gt, n=len(g), k=k))
print(len(TR), 'traces', sum(t['k'] is None for t in TR), 'thinking', sum(t['k'] is not None for t in TR), 'closed', flush=True)
def hazard(tv, ti, zt):
    oth = tv.masked_fill(ti == THINK, -1e30).sort(-1, descending=True)[0][:, :20]
    cand = torch.cat([oth, zt[:, None]], 1); order = cand.argsort(-1, descending=True)[:, :20]
    vals = cand.gather(1, order); pr = vals.softmax(-1); keep = (pr.cumsum(-1) - pr) < 0.95
    return ((order == 20) * keep * pr).sum(-1) / (pr * keep).sum(-1)
def scan(h, g, hL):
    mask = torch.zeros(E.shape[0], dtype=torch.bool, device=dev); tv, ti, zt, kl, ag = [], [], [], [], []
    for s in range(0, len(g), 256):
        z = h[s:s + 256] @ E.T; zl = z if hL is None else hL[s:s + 256] @ E.T
        la, ll = z.log_softmax(-1), zl.log_softmax(-1); kl.append((ll.exp() * (ll - la)).sum(-1)); ag.append(z.argmax(-1) == zl.argmax(-1))
        n = z.shape[0]; zp = z - PP * mask[None]; gi = g[s:s + n]; jj, ii = torch.tril_indices(n, n, -1, device=dev)
        zp[jj, gi[ii]] = z[jj, gi[ii]] - PP; v, i = zp.topk(21, dim=-1); tv.extend(v); ti.extend(i); zt.extend(z[:, THINK]); mask[gi] = True
    return torch.stack(tv), torch.stack(ti), torch.stack(zt), torch.cat(kl), torch.cat(ag)
HL = {}; RES = {}
def setw(ws):
    for l in LAY: lm.layers[l].mlp.down_proj.weight.data.copy_(ws[l].float().to(dev))
for arm in ['leader', 'parent'] + [a for a in ARMS if a not in ('leader', 'parent')]:
    setw(ARMS[arm]); loop = {l: [] for l in LAY}; gaps = {}; KLt, AGt, KLa, AGa, p32, pL = [], [], [], [], [], []
    for t in TR:
        with torch.inference_mode():
            model(input_ids=t['ids'], logits_to_keep=1); a, b = t['P'] - 1, t['P'] - 1 + t['n']; h = cap['h'][0, a:b].float()
            pj = {l: (cap[l][0, a:b].float() @ U[l].to(dev)) for l in LAY}
            if arm == 'leader': HL[t['name']] = h
            tv, ti, zt, kl, ag = scan(h, t['g'], None if arm == 'leader' else HL[t['name']])
        if t['k'] is None:
            for l in LAY: loop[l].append(pj[l])
            gaps.setdefault(t['fam'], []).append(tv[:, 19] - zt); KLt.append(kl); AGt.append(ag)
            p = hazard(tv, ti, zt); n = t['n']; S = torch.log1p(-p.clamp(max=1 - 1e-9)).cumsum(0).exp(); pt = p[-1024:].mean().item()
            p32.append(1 - S[-1].item() * math.exp(-(32768 - n) * pt)); pL.append(1 - S[-1].item())
        else: KLa.append(kl[t['k'] + 1:]); AGa.append(ag[t['k'] + 1:].float())
    th = [t for t in TR if t['k'] is None]; r = {}
    for l in LAY:
        r[f'loop{l}_mean'] = torch.cat(loop[l]).mean().item(); q4 = [(x[-(len(x) // 4):].mean().item(), x[:len(x) // 4].mean().item()) for x, t in zip(loop[l], th) if t['n'] >= 4000]
        r[f'loop{l}_last_over_first_4k'] = sum(a for a, _ in q4) / sum(b for _, b in q4)
    for fam, v in gaps.items(): c = torch.cat(v).float().cpu(); r[f'gap_{fam}'] = dict(median=c.median().item(), p10=c.quantile(0.1).item(), n=len(c))
    r.update(KL_think=torch.cat(KLt).mean().item(), top1_think=torch.cat(AGt).float().mean().item(), KL_answer=torch.cat(KLa).mean().item(), top1_answer=torch.cat(AGa).mean().item(),
             P32k_mean=sum(p32) / len(p32), P32k_max=max(p32), P_close_within_trace_mean=sum(pL) / len(pL))
    r['bf16_rel_change_vs_parent'] = {l: ((ARMS[arm][l].float() - Wp[l].float()).norm() / Wp[l].float().norm()).item() for l in LAY}
    r['bf16_rel_change_vs_leader'] = {l: ((ARMS[arm][l].float() - Wl[l].float()).norm() / Wl[l].float().norm()).item() for l in LAY}
    RES[arm] = r; print(arm, json.dumps(r, default=float), round(time.time() - T0), 's', flush=True)
for l in LAY:  # orient u so the original model writes a positive mean component (see header)
    if RES['parent'][f'loop{l}_mean'] < 0:
        U[l] = -U[l]
        for r in RES.values(): r[f'loop{l}_mean'] = -r[f'loop{l}_mean']
json.dump(dict(step1={k: v for k, v in S1.items()}, u={l: U[l].tolist() for l in LAY}, arms=RES, top64_rule=top64, wall=time.time() - T0), open('dose-results.json', 'w'), default=float)
print('\n%-8s %9s %9s %7s %7s %7s %7s %7s %7s %7s %7s' % ('arm', 'L15loop', 'L19loop', 'r15', 'r19', 'KLthk', 'KLans', 'top1t', 'top1a', 'P32k', 'gap median/p10 (arc/gsm8k/math500)'))
for a, r in RES.items():
    print('%-8s %9.3f %9.3f %7.2f %7.2f %7.4f %7.4f %7.4f %7.4f %7.4f' % (a, r['loop15_mean'], r['loop19_mean'], r['loop15_last_over_first_4k'], r['loop19_last_over_first_4k'], r['KL_think'], r['KL_answer'], r['top1_think'], r['top1_answer'], r['P32k_mean']),
          [f"{r[f'gap_{f}']['median']:.2f}/{r[f'gap_{f}']['p10']:.2f}" if f'gap_{f}' in r else f'{f}: no thinking traces' for f in ('arc', 'gsm8k', 'math500')])
print('wall', round(time.time() - T0), 's')
