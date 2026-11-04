import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from delta_cli import local, main
from delta_cli.artifacts import file_hash
from delta_cli.grid import DEFAULT_GRID, grid_hash


@pytest.fixture
def workspace(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    save_file({'layer.weight': torch.tensor([1., 2.])}, str(model / 'model.safetensors'))
    (model / 'config.json').write_text('{}')
    metadata = {'id': '00000000-0000-0000-0000-000000000002', 'model_id': 'tiny', 'content_hash': 'a' * 64, 'files': [
        {'name': p.name, 'size_bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in model.iterdir()]}
    (model / 'delta-model.json').write_text(json.dumps(metadata))
    data = {'a': [{'id': 'a1', 'prompt': 'A', 'answer': 'yes', 'samples': 2}],
            'b': [{'id': 'b1', 'prompt': 'B', 'answer': 'yes'}, {'id': 'b2', 'prompt': 'C', 'answer': 'yes'}]}
    (tmp_path / 'validation.json').write_text(json.dumps(data))
    config = {'id': 'tiny-v1', 'model_id': 'tiny', 'generation': {'max_tokens': 20, 'temperature': .6},
              'validation_names': list(data), 'validation_sha256': file_hash(tmp_path / 'validation.json')}
    (tmp_path / 'evaluation.json').write_text(json.dumps(config))
    (tmp_path / 'delta-workspace.json').write_text(json.dumps({'model_id': 'tiny', 'config_id': 'tiny-v1', 'parent_version_id': '00000000-0000-0000-0000-000000000002'}))
    return tmp_path, metadata, config


@pytest.fixture
def calls(monkeypatch):
    seen = []
    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, *args):
            pass
        def generate(self, prompt, generation, seed):
            seen.append(generation)
            return {'text': 'yes' if generation.get('temperature', .6) <= .6 else 'no', 'tokens': 2,
                    'generation_seconds': .1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', FakeRunner)
    return seen


def test_default_grid_is_preregistered():
    assert DEFAULT_GRID['default'] == {}
    canonical = json.dumps(DEFAULT_GRID, sort_keys=True, separators=(',', ':'))
    assert grid_hash(DEFAULT_GRID) == hashlib.sha256(canonical.encode()).hexdigest()


def test_grid_reports_and_summary(workspace, calls):
    root, _, _ = workspace
    out = root / 'grid-results'
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--output-dir', str(out)])
    assert result.exit_code == 0, result.output
    for name in DEFAULT_GRID:
        report = json.loads((out / f'{name}.json').read_text())
        assert report['grid_cell'] == name and report['grid_overrides'] == DEFAULT_GRID[name]
        assert (out / f'{name}.generations.jsonl').samefile(report['generations_file'])
        assert report['complete'] and report['validation_sha256'] and 'validation_score' in report
    summary = json.loads((out / 'grid-summary.json').read_text())
    assert summary['best_cell'] == 'default' and summary['default_cell'] == 'default'
    assert summary['best_minus_default'] == 0
    assert summary['preregistered'] is True and summary['grid_sha256'] == grid_hash(DEFAULT_GRID)
    assert [c['name'] for c in summary['cells']] == list(DEFAULT_GRID)
    assert {'local_only', 'model_id', 'config_id', 'source_version_id', 'artifact', 'selected_benchmarks', 'note'} <= set(summary)
    assert 'Best: default' in result.output


def test_custom_grid_rejects_unknown_key(workspace, calls):
    root, _, _ = workspace
    (root / 'bad.json').write_text(json.dumps({'x': {'beam': 1}}))
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'bad.json')])
    assert result.exit_code == 1 and "'beam'" in result.output and "'x'" in result.output
    assert not calls


def test_cells_subset_and_existing_output_protection(workspace, calls):
    root, _, _ = workspace
    args = ['grid', '--workspace', str(root), '--cells', 'temp-0.8', '--output-dir', str(root / 'g')]
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (root / 'g').glob('*.json')) == ['default.json', 'grid-summary.json', 'temp-0.8.json']
    summary = json.loads((root / 'g/grid-summary.json').read_text())
    assert summary['best_cell'] == 'default' and summary['best_minus_default'] == 0
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 1 and 'already exists' in result.output


def test_max_new_tokens_reaches_every_cell(workspace, calls):
    root, _, _ = workspace
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--limit', '1', '--max-new-tokens', '4',
                                           '--output-dir', str(root / 'g')])
    assert result.exit_code == 0, result.output
    assert calls and all(g['max_tokens'] == 4 for g in calls)
    assert json.loads((root / 'g/temp-0.6.json').read_text())['generation']['max_tokens'] == 4


def test_non_default_cell_can_win(workspace, monkeypatch):
    root, _, _ = workspace
    class HotRunner:
        details = {'device': 'cpu'}
        def __init__(self, *args):
            pass
        def generate(self, prompt, generation, seed):
            return {'text': 'yes' if generation.get('temperature', .6) >= .8 else 'no', 'tokens': 1,
                    'generation_seconds': .1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', HotRunner)
    (root / 'grid.json').write_text(json.dumps({'hot': {'temperature': .8}}))
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json'), '--output-dir', str(root / 'g')])
    assert result.exit_code == 0, result.output
    summary = json.loads((root / 'g/grid-summary.json').read_text())
    assert summary['best_cell'] == 'hot' and summary['best_minus_default'] == 100 and summary['preregistered'] is False
    assert [c['name'] for c in summary['cells']] == ['default', 'hot']
    assert 'Best: hot 100.0 (+100.0 over default)' in result.output
