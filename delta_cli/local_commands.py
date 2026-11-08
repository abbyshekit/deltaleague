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
    return {**local.generation_settings(config), **{k: v for k, v in overrides if v is not None}}


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
    report = {**details, **result, 'generation': generation, 'generation_overridden': generation != local.generation_settings(config),
              'complete': True, 'full_validation_set': full, 'selected_benchmarks': names,
              'examples_per_benchmark_limit': limit, 'validation_sha256': config['validation_sha256'],
              'generations_file': str(local.generations_path(recorded_output or output)), 'note': NOTE, **extra}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    return report



def fail(exc):
    if isinstance(exc, ImportError):
        typer.echo('Local evaluation dependencies are missing. Install the CLI with pip install ".[local]".', err=True)
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
                   'generation_overridden': generation != local.generation_settings(config),
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
             samples: int | None = typer.Option(None, min=1, max=16),
             validation_file: Path | None = typer.Option(None, '--validation'),
             output: Path = Path('local-results.json'),
             endpoint: str | None = None, served_model: str = 'candidate',
             content_hash: str | None = None, tokenizer: Path | None = None,
             concurrency: int = typer.Option(32, min=1, max=256)):
    """Evaluate public validation; advisory only. Pinned suites need --endpoint and --content-hash.

    Limits, checkpoint selectors and generation overrides apply only to legacy configurations.
    """
    try:
        root, _, config, _ = local.read_workspace(workspace)
        if validation_file and (endpoint or content_hash or tokenizer):
            raise ValueError('--validation uses local inference; do not combine it with endpoint options.')
        if config.get('protocol_version') == 'delta-public-v1' and validation_file is None:
            if samples is not None:
                raise ValueError('--samples is only supported for legacy/custom local evaluations, not the pinned public suite.')
            if not endpoint or not content_hash:
                raise ValueError('This configuration uses the pinned public evaluator. Supply --endpoint and '
                                 '--content-hash for your served checkpoint, or use `delta public-eval run --help`. '
                                 'See PUBLIC_EVALUATION.md for the isolated runtime setup.')
            if version or weights or patch or benchmark or limit or max_new_tokens is not None or thinking is not None:
                raise ValueError('The pinned evaluator uses the complete frozen suite and its own generation settings. '
                                 'Serve your chosen checkpoint first; use --content-hash to identify it.')
            if local.file_hash(root / 'validation.json') != config['validation_sha256']:
                raise ValueError('Public validation data changed; initialize a fresh workspace.')
            from .public_eval.cli import run_command
            run_command(endpoint=endpoint, model=served_model, tokenizer=tokenizer or root / 'model',
                        content_hash=content_hash, manifest=root / 'validation.json', concurrency=concurrency,
                        output=Path('public-results') if output == Path('local-results.json') else output,
                        evaluation_config=root / 'evaluation.json')
            return
        if endpoint or content_hash or tokenizer:
            raise ValueError('Endpoint evaluation requires a delta-public-v1 configuration.')
        path = (validation_file or root / 'validation.json').resolve()
        from .public_eval.custom import check_runtime
        check_runtime()
        tasks, names, full, sha = local.public_examples(root, config, benchmark, limit,
                                                      validation_file=path if validation_file else None, samples=samples)
        output = output.resolve()
        generations = local.generations_path(output)
        if output.exists() or generations.exists():
            raise ValueError('Output already exists. Choose a new --output path to preserve previous results.')
        output.parent.mkdir(parents=True, exist_ok=True)
        budget = max_new_tokens or local.generation_settings(config).get('max_tokens', 32768)
        typer.echo(f'{len(tasks)} generations; up to {len(tasks) * budget:,} output tokens. Use --limit and --max-new-tokens for a quick check.', err=True)
        with experiment(workspace, version, weights, patch, device, max_new_tokens, thinking) as (runner, details):
            report = write_report(runner, details, config, tasks, names, full, limit, details['generation'], output,
                lambda index, total, name: typer.echo(f'[{index}/{total}] {name}', err=True),
                validation_sha256=sha, validation_file=str(path),
                validation_kind='custom' if validation_file else 'workspace', samples_override=samples)
            typer.echo(json.dumps(report, indent=2))
            typer.echo(f'Saved {output}', err=True)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        fail(exc)
