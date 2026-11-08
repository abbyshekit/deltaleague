"""Pre-registered decoding grid: one model load, one evaluation per generation setting."""
import hashlib
import json
import math
import re
from pathlib import Path

import typer

from . import local
from .local_commands import NOTE, budget_line, experiment, fail, generation_for, write_report

RULES = {'enable_thinking': (bool, 'a boolean'), 'max_tokens': (int, 'a positive integer'), 'top_k': (int, 'a positive integer'),
         'temperature': (float, 'a non-negative number'), 'top_p': (float, 'a number from 0 to 1'),
         'presence_penalty': (float, 'a non-negative number')}
DEFAULT_GRID = {
    'default': {},
    'temp-0.6': {'temperature': 0.6},
    'temp-0.8': {'temperature': 0.8},
    'presence-0.5': {'presence_penalty': 0.5},
    'no-thinking': {'enable_thinking': False},
    'temp-0.6-no-thinking': {'temperature': 0.6, 'enable_thinking': False},
}


def grid_hash(grid):
    return hashlib.sha256(json.dumps(grid, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_value(name, key, value):
    kind, label = RULES[key]
    if kind is bool:
        ok = isinstance(value, bool)
    elif kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 10**9
    else:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and (isinstance(value, int) or math.isfinite(value))
        ok = ok and 0 <= value <= (1 if key == 'top_p' else 10**9)
    if not ok:
        raise ValueError(f"Generation key '{key}' in cell '{name}' must be {label}.")


def load_grid(path=None):
    if path is None:
        return DEFAULT_GRID
    grid = json.loads(Path(path).read_text())
    if not isinstance(grid, dict) or not grid:
        raise ValueError('Grid file must be a non-empty JSON object of {cell: {overrides}}.')
    seen = {}
    for name, overrides in grid.items():
        if not re.fullmatch(r'[A-Za-z0-9._-]+', name) or name.startswith('.'):
            raise ValueError(f"Invalid cell name '{name}': use letters, digits, '.', '_' and '-', not starting with '.'.")
        if (other := seen.setdefault(name.lower(), name)) != name:
            raise ValueError(f"Cell names must differ case-insensitively: '{other}' and '{name}'.")
        if name.lower() == 'grid-summary':
            raise ValueError("Cell name 'grid-summary' is reserved for the summary file.")
        if not isinstance(overrides, dict):
            raise ValueError(f"Cell '{name}' must be an object of generation overrides.")
        if name == 'default' and overrides:
            raise ValueError("Cell 'default' must have empty overrides; it is the baseline.")
        for key, value in overrides.items():
            if key not in RULES:
                raise ValueError(f"Unknown generation key '{key}' in cell '{name}'; allowed: {', '.join(sorted(RULES))}.")
            check_value(name, key, value)
    if not any(not overrides for overrides in grid.values()):
        if 'default' in seen:
            raise ValueError('Cell name conflicts with the implicit default baseline.')
        return {'default': {}, **grid}
    return grid


def select(grid, cells):
    """The baseline cell name and the cells to run (always including the baseline)."""
    default = next((name for name, overrides in grid.items() if not overrides), None)
    if default is None:
        raise ValueError('Grid has no baseline cell with empty overrides.')
    wanted = {c.strip() for c in (cells or '').split(',') if c.strip()}
    if unknown := wanted - set(grid):
        raise ValueError('Unknown cell(s): ' + ', '.join(sorted(unknown)) + '; choose from: ' + ', '.join(grid))
    return default, [name for name in grid if not wanted or name in wanted or name == default]


def best_cell(cells, default):
    top = max(cell['validation_score'] for cell in cells)
    tied = [cell['name'] for cell in cells if cell['validation_score'] == top]
    return default if default in tied else tied[0]


def grid(workspace: Path = Path('.'), version: str | None = None,
         weights: Path | None = None, patch: Path | None = None, device: str = 'auto',
         benchmark: str | None = None, limit: int | None = typer.Option(None, min=1),
         max_new_tokens: int | None = typer.Option(None, min=1),
         thinking: bool | None = typer.Option(None, '--thinking/--no-thinking'),
         validation_file: Path | None = typer.Option(None, '--validation'),
         samples: int | None = typer.Option(None, min=1, max=16),
         grid_file: Path | None = typer.Option(None, '--grid'),
         cells: str | None = None, output_dir: Path = Path('grid-results')):
    """Evaluate one checkpoint under each decoding setting of a pre-registered grid. Local control, never an official score."""
    try:
        root, _, config, _ = local.read_workspace(workspace)
        if config.get('protocol_version') == 'delta-public-v1' and validation_file is None:
            raise ValueError('Decoding grids require legacy workspace data or an explicit --validation file. '
                             'Use delta public-eval for the unchanged pinned public suite.')
        from .public_eval.custom import check_runtime
        check_runtime()
        validation_path = (validation_file or root / 'validation.json').resolve()
        tasks, names, full, validation_sha = local.public_examples(root, config, benchmark, limit,
            validation_file=validation_path if validation_file else None, samples=samples)
        cell_grid = load_grid(grid_file)
        default, selected = select(cell_grid, cells)
        output_dir = output_dir.resolve()
        partial = output_dir.with_name(output_dir.name + '.partial')
        if output_dir.exists() or partial.exists():
            raise ValueError(f'Output already exists: {output_dir} or {partial}. Choose a new --output-dir to preserve previous results.')
        effective = [({**generation_for(config, max_new_tokens, thinking), **cell_grid[name]}, len(tasks)) for name in selected]
        limit_len = local.generation_settings(config).get('max_model_len', 34816)
        for name, (generation, _) in zip(selected, effective):
            if (tokens := generation.get('max_tokens', local.MAX_TOKENS)) >= limit_len:
                raise ValueError(f"Cell '{name}' sets max_tokens {tokens}, at or above max_model_len {limit_len}.")
        grid_sha256 = grid_hash(cell_grid)
        preregistered = cell_grid == DEFAULT_GRID
        budget_line(effective)
        typer.echo(f"Grid {grid_sha256}{', pre-registered default' if preregistered else ''}; cells: {', '.join(selected)}", err=True)
        with experiment(workspace, version, weights, patch, device, max_new_tokens, thinking) as (runner, details):
            partial.mkdir(parents=True)
            rows = []
            for number, name in enumerate(selected, 1):
                def progress(index, total, bench, label=f'[cell {number}/{len(selected)} {name}]'):
                    typer.echo(f'{label} [{index}/{total}] {bench}', err=True)
                report = write_report(runner, details, config, tasks, names, full, limit, {**details['generation'], **cell_grid[name]},
                    partial / f'{name}.json', progress, recorded_output=output_dir / f'{name}.json',
                    grid_cell=name, grid_overrides=cell_grid[name], validation_sha256=validation_sha,
                    validation_file=str(validation_path), validation_kind='custom' if validation_file else 'workspace',
                    samples_override=samples)
                rows.append({'name': name, 'overrides': cell_grid[name], 'report': str(output_dir / f'{name}.json'),
                             **{k: report[k] for k in ('validation_score', 'validation_benchmarks', 'truncation_rate', 'output_tokens')}})
            best = best_cell(rows, default)
            score = {row['name']: row['validation_score'] for row in rows}
            summary = {'local_only': True, **{k: details[k] for k in ('model_id', 'config_id', 'source_version_id', 'artifact')},
                       'grid_sha256': grid_sha256, 'preregistered': preregistered, 'cells_requested': selected, 'cells': rows,
                       'default_cell': default, 'best_cell': best, 'best_minus_default': score[best] - score[default],
                       'selected_benchmarks': names, 'examples_per_benchmark_limit': limit,
                       'validation_sha256': validation_sha, 'validation_file': str(validation_path),
                       'samples_override': samples,
                       'effective_generation': {name: generation for name, (generation, _) in zip(selected, effective)},
                       'note': NOTE + ' Best-cell selection uses the same data; confirm gains on separate data.'}
            with (partial / 'grid-summary.json').open('x') as stream:
                json.dump(summary, stream, indent=2)
        partial.rename(output_dir)
        for row in rows:
            typer.echo(f"{row['name']:<24}{row['validation_score']:6.1f}  {row['truncation_rate'] * 100:5.1f}% truncated  {row['output_tokens']:>9,} tokens")
        typer.echo(f"Best: {best} {score[best]:.1f} ({summary['best_minus_default']:+.1f} over {default}). Compare a candidate with: delta compare {output_dir / (best + '.json')} candidate.json")
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        fail(exc)
