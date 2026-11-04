"""Pre-registered decoding grid: one model load, one evaluation per generation setting."""
import hashlib
import json
import re
from pathlib import Path

import typer

from . import local
from .local_commands import experiment, fail

ALLOWED_KEYS = {'temperature', 'top_p', 'top_k', 'presence_penalty', 'max_tokens', 'enable_thinking'}
DEFAULT_GRID = {
    'default': {},
    'temp-0.6': {'temperature': 0.6},
    'temp-0.8': {'temperature': 0.8},
    'presence-0.5': {'presence_penalty': 0.5},
    'no-thinking': {'enable_thinking': False},
    'temp-0.6-no-thinking': {'temperature': 0.6, 'enable_thinking': False},
}
NOTE = 'Advisory local score. Hardware/backend differences affect results; this is not a verified leaderboard score.'


def grid_hash(grid):
    return hashlib.sha256(json.dumps(grid, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_grid(path):
    grid = json.loads(Path(path).read_text())
    if not isinstance(grid, dict) or not grid:
        raise ValueError('Grid file must be a non-empty JSON object of {cell: {overrides}}.')
    for name, overrides in grid.items():
        if not re.fullmatch(r'[A-Za-z0-9._-]+', name):
            raise ValueError(f"Invalid cell name '{name}': use letters, digits, '.', '_' and '-'.")
        if not isinstance(overrides, dict):
            raise ValueError(f"Cell '{name}' must be an object of generation overrides.")
        for key, value in overrides.items():
            if key not in ALLOWED_KEYS:
                raise ValueError(f"Unknown generation key '{key}' in cell '{name}'; allowed: {', '.join(sorted(ALLOWED_KEYS))}.")
            if not isinstance(value, (int, float, bool)):
                raise ValueError(f"Generation key '{key}' in cell '{name}' must be a number or boolean.")
    if not any(not overrides for overrides in grid.values()):
        grid = {'default': {}, **grid}
    return grid


def default_name(grid):
    return next(name for name, overrides in grid.items() if not overrides)


def best_cell(cells, default):
    top = max(cell['validation_score'] for cell in cells)
    tied = [cell['name'] for cell in cells if cell['validation_score'] == top]
    return default if default in tied else tied[0]


def grid(workspace: Path = Path('.'), version: str | None = None,
         weights: Path | None = None, patch: Path | None = None, device: str = 'auto',
         benchmark: str | None = None, limit: int | None = typer.Option(None, min=1),
         max_new_tokens: int | None = typer.Option(None, min=1),
         thinking: bool | None = typer.Option(None, '--thinking/--no-thinking'),
         grid_file: Path | None = typer.Option(None, '--grid'),
         cells: str | None = None, output_dir: Path = Path('grid-results')):
    """Evaluate one checkpoint under each decoding setting of a pre-registered grid. Local control, never an official score."""
    try:
        root, _, config, _ = local.read_workspace(workspace)
        tasks, names, full = local.public_examples(root, config, benchmark, limit)
        cell_grid = load_grid(grid_file) if grid_file else DEFAULT_GRID
        default = default_name(cell_grid)
        selected = list(cell_grid)
        if cells:
            wanted = {c.strip() for c in cells.split(',') if c.strip()}
            if wanted - set(cell_grid):
                raise ValueError('Unknown cell(s): ' + ', '.join(sorted(wanted - set(cell_grid))) + '; choose from: ' + ', '.join(cell_grid))
            selected = [name for name in cell_grid if name in wanted or name == default]
        output_dir = output_dir.resolve()
        files = [output_dir / 'grid-summary.json'] + [output_dir / f'{name}{suffix}' for name in selected for suffix in ('.json', '.generations.jsonl')]
        if any(path.exists() for path in files):
            raise ValueError('Output already exists in the grid output directory. Choose a new --output-dir to preserve previous results.')
        base = {**config['generation'], **{k: v for k, v in (('max_tokens', max_new_tokens), ('enable_thinking', thinking)) if v is not None}}
        budget = sum(len(tasks) * {**base, **cell_grid[name]}.get('max_tokens', 32768) for name in selected)
        typer.echo(f'{len(selected)} cells × {len(tasks)} generations; up to {budget:,} output tokens. Use --limit, --cells and --max-new-tokens for a quick check.', err=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with experiment(workspace, version, weights, patch, device, max_new_tokens, thinking) as (runner, details):
            for number, name in enumerate(selected, 1):
                generation = {**details['generation'], **cell_grid[name]}
                path, generations = output_dir / f'{name}.json', output_dir / f'{name}.generations.jsonl'
                with generations.open('x') as stream:
                    result = local.evaluate(runner, tasks, generation, stream,
                        lambda index, total, bench, number=number, name=name: typer.echo(f'[cell {number}/{len(selected)} {name}] [{index}/{total}] {bench}', err=True))
                report = {**details, **result, 'generation': generation, 'generation_overridden': generation != config['generation'],
                          'complete': True, 'full_validation_set': full, 'selected_benchmarks': names,
                          'examples_per_benchmark_limit': limit, 'validation_sha256': config['validation_sha256'],
                          'generations_file': str(generations), 'note': NOTE,
                          'grid_cell': name, 'grid_overrides': cell_grid[name]}
                with path.open('x') as stream:
                    json.dump(report, stream, indent=2)
                rows.append({'name': name, 'overrides': cell_grid[name], 'validation_score': result['validation_score'],
                             'validation_benchmarks': result['validation_benchmarks'], 'truncation_rate': result['truncation_rate'],
                             'output_tokens': result['output_tokens'], 'report': str(path)})
            best = best_cell(rows, default)
            score = {row['name']: row['validation_score'] for row in rows}
            summary = {'local_only': True, 'model_id': details['model_id'], 'config_id': details['config_id'],
                       'source_version_id': details['source_version_id'], 'artifact': details['artifact'],
                       'grid_sha256': grid_hash(cell_grid), 'preregistered': cell_grid == DEFAULT_GRID, 'cells': rows,
                       'default_cell': default, 'best_cell': best, 'best_minus_default': score[best] - score[default],
                       'selected_benchmarks': names, 'examples_per_benchmark_limit': limit, 'note': NOTE}
            with (output_dir / 'grid-summary.json').open('x') as stream:
                json.dump(summary, stream, indent=2)
        for row in rows:
            typer.echo(f"{row['name']:<24}{row['validation_score']:6.1f}  {row['truncation_rate'] * 100:5.1f}% truncated  {row['output_tokens']:>9,} tokens")
        gain = summary['best_minus_default']
        typer.echo(f"Best: {best} {score[best]:.1f} ({gain:+.1f} over default). Compare a candidate with: delta compare {output_dir / (best + '.json')} candidate.json")
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        fail(exc)
