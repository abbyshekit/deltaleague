"""Deterministic checks on the workflow's accepted items, then seal a held-out set in the validation.json format.

Vetoes (each recorded): a GSM program that does not return the stated answer; a MATH answer math-verify cannot parse;
a solver answer math-verify does not accept against the writer's; a 10-gram overlap with any public test item (ARC: on the stem only) or a public ARC question contained in the item;
a duplicate inside the set; with --exclude, a 10-gram overlap with or a duplicate of any item of an earlier set (validation.json format).
Usage: build_heldout.py [--exclude EARLIER.json ...] RESULT.json [RESULT2.json ...] OUT_DIR
Private fields the workflow attaches to an item (solutions, programs, solver answers, solver_flags) pass through to the private sidecar.
"""
import argparse
import hashlib
import json
import ast
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

ap = argparse.ArgumentParser(usage='build_heldout.py [--exclude EARLIER.json ...] RESULT.json [RESULT2.json ...] OUT_DIR')
ap.add_argument('--exclude', action='append', default=[], metavar='PATH', help='earlier set in validation.json format; repeatable')
ap.add_argument('paths', nargs='+', help='workflow result JSONs followed by OUT_DIR')
cli = ap.parse_intermixed_args()
if len(cli.paths) < 2:
    ap.error('need at least one RESULT.json and OUT_DIR')
inputs, out = cli.paths[:-1], Path(cli.paths[-1]); out.mkdir(parents=True, exist_ok=True)
merged = {}
for path in inputs:
    for fam in json.loads(Path(path).read_text(encoding='utf-8'))['families']:
        m = merged.setdefault(fam['name'], {'name': fam['name'], 'scorer': fam['scorer'], 'accepted': [], 'rejected': [], 'written': 0, 'rounds': 0})
        m['accepted'] += fam['accepted']; m['rejected'] += fam['rejected']; m['written'] += fam['written']; m['rounds'] += fam['rounds']
result = {'families': list(merged.values())}
public_dir = Path(__file__).parent / 'public'
PUBLIC = {}
for name, field in (('arc_challenge', 'question'), ('gsm8k', 'question'), ('math500', 'problem')):
    PUBLIC[name] = [row[field] for row in json.loads((public_dir / f'{name}.json').read_text(encoding='utf-8'))]


def words(text):
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).split()


def grams(text, n=10):
    w = words(text)
    return {' '.join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


PUBLIC_GRAMS = {name: set().union(*(grams(t) for t in texts)) for name, texts in PUBLIC.items()}
ALL_PUBLIC_GRAMS = set().union(*PUBLIC_GRAMS.values())
ARC_NORM = {' '.join(words(q)) for q in PUBLIC['arc_challenge']}
ARC_NORM = {f' {q} ' for q in ARC_NORM if len(q.split()) >= 4}
EXCLUDE_GRAMS, EXCLUDE_KEYS, EXCLUDE_COUNTS = set(), set(), {}
for path in cli.exclude:
    for fam_name, rows in json.loads(Path(path).read_text(encoding='utf-8')).items():
        for row in rows:
            ex_body = row['prompt'].split('\nPlease reason step by step')[0]
            EXCLUDE_GRAMS |= grams(ex_body); EXCLUDE_KEYS.add(' '.join(words(ex_body))[:90])
            EXCLUDE_COUNTS[fam_name] = EXCLUDE_COUNTS.get(fam_name, 0) + 1


def mv_equal(gold, pred):
    cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
    g, p = parse(f'${gold}$', extraction_config=cfg), parse(f'${pred}$', extraction_config=cfg)
    return bool(g) and bool(p) and verify(g, p)


BAD_CALLS = {'open', 'exec', 'eval', '__import__', 'compile', 'input', 'breakpoint'}
BAD_BASES = {'os', 'sys', 'subprocess', 'socket', 'builtins'}
LIMITS = ((resource.RLIMIT_AS, 512 * 2**20), (resource.RLIMIT_CPU, 5), (resource.RLIMIT_FSIZE, 2**20), (resource.RLIMIT_NOFILE, 16))


def veto_program(program):
    try:
        tree = ast.parse(program)
    except SyntaxError as e:
        return f'program syntax error: {e.msg}'
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name != 'math' for a in node.names):
            return 'program imports ' + ', '.join(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom) and (node.module != 'math' or node.level):
            return 'program imports ' + '.' * node.level + (node.module or '')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BAD_CALLS:
            return f'program calls {node.func.id}'
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in BAD_BASES:
            return f'program uses {node.value.id}.{node.attr}'
    return None


def limit_child():
    for res, val in LIMITS:
        try:
            resource.setrlimit(res, (val, val))
        except (ValueError, OSError):
            pass


def run_program(program, numeric_answer):
    if not program.strip():
        return 'no program'
    if (veto := veto_program(program)):
        return veto
    code = program + '\n\nimport json\nprint(json.dumps(solve()))\n'
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code); path = f.name
    try:
        with tempfile.TemporaryDirectory() as cwd, open(Path(cwd) / 'out.txt', 'wb+') as so, open(Path(cwd) / 'err.txt', 'wb+') as se:
            try:
                r = subprocess.run([sys.executable, '-I', path], stdout=so, stderr=se, timeout=10, cwd=cwd,
                                   env={'PATH': '/usr/bin:/bin'}, preexec_fn=limit_child)
            except subprocess.TimeoutExpired:
                return 'program timeout'
            so.seek(0); se.seek(0)
            out, err = so.read(4096).decode('utf-8', 'replace').strip(), se.read(4096).decode('utf-8', 'replace').strip()
    finally:
        Path(path).unlink(missing_ok=True)
    if r.returncode != 0:
        return 'program error: ' + err.splitlines()[-1][:120] if err else 'program error'
    try:
        value = float(json.loads(out))
    except Exception:
        return 'program output not numeric: ' + out[:60]
    return None if abs(value - numeric_answer) < 1e-6 else f'program returns {value}, item says {numeric_answer}'


sealed, private, report = {}, {}, {'families': {}}
for fam in result['families']:
    name, kept, vetoed, seen = fam['name'], [], [], set()
    for it in fam['accepted']:
        body = it['prompt'].split('\nPlease reason step by step')[0]
        reasons = []
        stem = re.split(r'^A\. ', body, maxsplit=1, flags=re.M)[0] if name == 'arc_fresh' else body
        hit = grams(stem) & ALL_PUBLIC_GRAMS
        if hit:
            reasons.append('10-gram overlap with a public item: ' + sorted(hit)[0][:80])
        norm = ' '.join(words(body))
        if name == 'arc_fresh':
            contained = [q for q in ARC_NORM if q in f' {norm} ']
            if contained:
                reasons.append('contains a public ARC question: ' + contained[0].strip()[:80])
        if EXCLUDE_GRAMS and (hit := grams(body) & EXCLUDE_GRAMS):
            reasons.append('10-gram overlap with an excluded item: ' + sorted(hit)[0][:80])
        k = norm[:90]
        if k in seen:
            reasons.append('duplicate body')
        if k in EXCLUDE_KEYS:
            reasons.append('duplicate of an excluded item')
        if name == 'gsm_fresh':
            try:
                numeric = float(str(it['answer']).replace(',', ''))
            except ValueError:
                numeric = None; reasons.append('answer not numeric')
            if numeric is not None:
                err = run_program(it['private'].get('program', ''), numeric)
                if err:
                    reasons.append(err)
        if name in ('gsm_fresh', 'math_fresh'):
            for who in ('solver_a', 'solver_b'):
                ans = it['private'].get(who)
                if ans and not mv_equal(it['answer'], ans):
                    reasons.append(f'math-verify rejects {who} answer {ans!r} against {it["answer"]!r}')
            if not mv_equal(it['answer'], it['answer']):
                reasons.append('math-verify cannot parse the answer')
        if name == 'arc_fresh':
            if it['answer'] not in {'A', 'B', 'C', 'D'}:
                reasons.append('answer is not a letter')
            opts = re.findall(r'^([A-D])\. ', body, flags=re.M)
            if opts != ['A', 'B', 'C', 'D']:
                reasons.append(f'options malformed: {opts}')
        if not reasons:
            seen.add(k)
        (vetoed if reasons else kept).append({**it, 'veto': reasons} if reasons else it)
    sealed[name] = [dict(it) for it in kept]
    report['families'][name] = {'workflow_accepted': len(fam['accepted']), 'workflow_rejected': len(fam['rejected']), 'written': fam['written'],
                                'rounds': fam['rounds'], 'vetoed': len(vetoed), 'sealed': len(kept),
                                'veto_reasons': sorted({r.split(':')[0] for v in vetoed for r in v['veto']}),
                                'letter_balance': {l: sum(1 for it in kept if it['answer'] == l) for l in 'ABCD'} if name == 'arc_fresh' else None}
    (out / f'vetoed-{name}.json').write_text(json.dumps(vetoed, indent=1), encoding='utf-8')
    (out / f'workflow-rejected-{name}.json').write_text(json.dumps(fam['rejected'], indent=1), encoding='utf-8')

def rotate_options(prompt, answer, shift):
    if '\nPlease reason step by step' not in prompt:
        return prompt, answer
    body, tail = prompt.split('\nPlease reason step by step', 1)
    lines = body.split('\n')
    opts = [l for l in lines if re.match(r'^[A-D]\. ', l)]
    if len(opts) != 4:
        return prompt, answer
    texts = [l[3:] for l in opts]
    rotated = texts[-shift:] + texts[:-shift] if shift else texts
    new_lines = [l for l in lines if not re.match(r'^[A-D]\. ', l)] + [f'{L}. {t}' for L, t in zip('ABCD', rotated)]
    return '\n'.join(new_lines) + '\nPlease reason step by step' + tail, 'ABCD'[('ABCD'.index(answer) + shift) % 4]


for name in sealed:
    for i, it in enumerate(sealed[name]):
        it['id'] = f'{name}-{i}'
        it['source_id'] = it['private'].get('source_id', '')
if 'arc_fresh' in sealed:
    items = sealed['arc_fresh']
    for i, it in enumerate(items):
        target = 'ABCD'[i % 4]
        shift = ('ABCD'.index(target) - 'ABCD'.index(it['answer'])) % 4
        it['private']['original_answer'], it['private']['rotation'] = it['answer'], shift
        it['prompt'], it['answer'] = rotate_options(it['prompt'], it['answer'], shift)
    report['families']['arc_fresh']['letter_balance_after_rotation'] = {l: sum(1 for it in items if it['answer'] == l) for l in 'ABCD'}
for name in sealed:
    private[name] = [{'id': it['id'], **it.pop('private')} for it in sealed[name]]
    for it in sealed[name]:
        it.pop('source_id', None)
text = json.dumps(sealed, ensure_ascii=False, indent=1)
data = text.encode('utf-8')
(out / 'heldout-v1.json').write_bytes(data)
(out / 'heldout-v1.private.json').write_bytes(json.dumps(private, ensure_ascii=False, indent=1).encode('utf-8'))
digest = hashlib.sha256(data).hexdigest()
(out / 'heldout-v1.sha256').write_text(digest + '  heldout-v1.json\n', encoding='utf-8')
report['sha256'] = digest
report['public_sets'] = {name: len(texts) for name, texts in PUBLIC.items()}
report['excluded'] = {'files': len(cli.exclude), 'items': sum(EXCLUDE_COUNTS.values()), 'by_family': EXCLUDE_COUNTS}
(out / 'report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
print(json.dumps(report, indent=1))
