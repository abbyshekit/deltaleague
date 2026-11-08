"""Download the public test splits the private test is drawn from, via the HF datasets-server rows API (no datasets lib needed)."""
import hashlib, json, sys, time, urllib.parse, urllib.request
from pathlib import Path
OUT = Path(__file__).parent if Path(__file__).parent.name == 'public' else Path(__file__).parent / 'public'
OUT.mkdir(exist_ok=True)
SETS = {
    'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge', 'test'),
    'gsm8k': ('openai/gsm8k', 'main', 'test'),
    'math500': ('HuggingFaceH4/MATH-500', 'default', 'test'),
}
for name, (ds, cfg, split) in SETS.items():
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({'dataset': ds, 'config': cfg, 'split': split, 'offset': offset, 'length': 100})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f'https://datasets-server.huggingface.co/rows?{q}', timeout=60) as r: page = json.load(r); break
            except Exception as exc:
                print(name, 'retry', attempt, type(exc).__name__, file=sys.stderr); time.sleep(3)
        else: raise SystemExit(f'{name}: gave up at offset {offset}')
        got = [x['row'] for x in page['rows']]; rows += got; offset += len(got)
        if len(got) < 100: break
    path = OUT / f'{name}.json'
    with open(path, 'w', encoding='utf-8') as f: json.dump(rows, f)
    print(name, len(rows), 'rows; sha256', hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
