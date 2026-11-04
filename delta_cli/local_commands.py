"""CLI wrappers keep optional inference dependencies out of upload/discovery commands."""
from contextlib import contextmanager
import hashlib
import json
import os
import sys
from pathlib import Path
import uuid

import typer

from . import local

NOTE = 'Advisory local score. Hardware/backend differences affect results; this is not a verified leaderboard score.'


def generation_for(config, max_new_tokens, thinking):
    """Config generation block with the CLI overrides applied."""
    overrides = (('max_tokens', max_new_tokens), ('enable_thinking', thinking))
    return {**config['generation'], **{k: v for k, v in overrides if v is not None}}


def budget_line(cells):
    """Echo the generation count and output-token ceiling for [(generation, task_count), ...]."""
    count = sum(n for _, n in cells)
    tokens = sum(n * g.get('max_tokens', local.MAX_TOKENS) for g, n in cells)
    typer.echo(f'{count} generations; up to {tokens:,} output tokens. Use --limit and --max-new-tokens for a quick check.', err=True)


def write_report(runner, details, config, tasks, names, full, limit, generation, output, progress, recorded_output=None, **extra):
    """Evaluate, then write the generations file and the report beside `output`."""
    generations = local.generations_path(output)
    with generations.open('x') as stream:
        result = local.evaluate(runner, tasks, generation, stream, progress)
    report = {**details, **result, 'generation': generation, 'generation_overridden': generation != config['generation'],
              'complete': True, 'full_validation_set': full, 'selected_benchmarks': names,
              'examples_per_benchmark_limit': limit, 'validation_sha256': config['validation_sha256'],
              'generations_file': str(local.generations_path(recorded_output or output)), 'note': NOTE, **extra}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    return report


def fail(exc):
    if isinstance(exc, ImportError):
        typer.echo('Local inference dependencies are missing. Install the CLI with pip install ".[local]"; IFEval also needs ".[ifeval]".', err=True)
    else:
        typer.echo(f'Local run failed: {exc}', err=True)
        typer.echo('For device/memory errors, try --device cpu or a smaller --max-new-tokens value.', err=True)
    raise typer.Exit(1)


@contextmanager
def experiment(workspace, version_id, weights, patch, device, max_new_tokens, thinking):
    from .main import API, call, pull_model
    workspace, identity, config, metadata = local.read_workspace(workspace)
    parent = workspace / 'model'
    if version_id:
        if identity.get("api", API) != API:
            raise ValueError("Workspace belongs to another API. Set delta --api WORKSPACE_API before the command.")
        version_id = str(uuid.UUID(version_id))
        remote = call('GET', f'model-versions/{version_id}')
        if remote['model_id'] != identity['model_id']:
            raise ValueError('Leaderboard version belongs to another model. Initialize that competition first.')
        parent = workspace / '.delta/models' / version_id
        metadata = pull_model(version_id, parent)
    generation = generation_for(config, max_new_tokens, thinking)
    if sys.platform == 'darwin' and device in {'auto', 'mps'}:
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    typer.echo('Checking model artifacts…', err=True)
    with local.model_directory(workspace, metadata, weights, patch, parent=parent) as (directory, artifact):
        typer.echo(f'Loading model on {device} (first load may take a while)…', err=True)
        runner = local.Runner(directory, device)
        details = {'local_only': True, 'model_id': identity['model_id'], 'config_id': config['id'],
                   'source_version_id': metadata['id'], 'source_content_hash': metadata['content_hash'],
                   'artifact': artifact, 'runtime': runner.details, 'generation': generation,
                   'generation_overridden': generation != config['generation'],
                   'evaluation_sha256': hashlib.sha256((workspace / 'evaluation.json').read_bytes()).hexdigest()}
        typer.echo(f"Running on {runner.details['device']}; local results are advisory.", err=True)
        yield runner, details


def run(prompt: str, workspace: Path = Path('.'), version: str | None = None,
        weights: Path | None = None, patch: Path | None = None, device: str = 'auto',
        max_new_tokens: int = typer.Option(256, min=1), seed: int = 1000,
        thinking: bool | None = typer.Option(None, '--thinking/--no-thinking'),
        json_output: bool = typer.Option(False, '--json')):
    """Run a prompt locally on the workspace parent, a leaderboard --version, or candidate weights."""
    try:
        with experiment(workspace, version, weights, patch, device, max_new_tokens, thinking) as (runner, details):
            result = runner.generate(prompt, details['generation'], seed)
            if json_output:
                typer.echo(json.dumps({**details, 'seed': seed, **result}, indent=2))
            else:
                typer.echo(result['text'])
                if result['truncated']:
                    typer.echo('Output reached the token limit. Increase --max-new-tokens for a complete answer.', err=True)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        fail(exc)


def evaluate(workspace: Path = Path('.'), version: str | None = None,
             weights: Path | None = None, patch: Path | None = None, device: str = 'auto',
             benchmark: str | None = None, limit: int | None = typer.Option(None, min=1),
             max_new_tokens: int | None = typer.Option(None, min=1),
             thinking: bool | None = typer.Option(None, '--thinking/--no-thinking'),
             output: Path = Path('local-results.json')):
    """Evaluate public validation. --limit caps examples per benchmark; output is never an official score."""
    try:
        root, _, config, _ = local.read_workspace(workspace)
        tasks, names, full = local.public_examples(root, config, benchmark, limit)
        output = output.resolve()
        if output.exists() or local.generations_path(output).exists():
            raise ValueError('Output already exists. Choose a new --output path to preserve previous results.')
        output.parent.mkdir(parents=True, exist_ok=True)
        budget_line([(generation_for(config, max_new_tokens, thinking), len(tasks))])
        with experiment(workspace, version, weights, patch, device, max_new_tokens, thinking) as (runner, details):
            report = write_report(runner, details, config, tasks, names, full, limit, details['generation'], output,
                lambda index, total, name: typer.echo(f'[{index}/{total}] {name}', err=True))
            typer.echo(json.dumps(report, indent=2))
            typer.echo(f'Saved {output}', err=True)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        fail(exc)
