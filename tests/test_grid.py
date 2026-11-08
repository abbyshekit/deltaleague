import json

import pytest
from typer.testing import CliRunner

from delta_cli import main
from delta_cli.grid import DEFAULT_GRID, best_cell, grid_hash, load_grid


@pytest.fixture
def calls(fake_runner):
    seen = []
    fake_runner(lambda g: 'yes' if g.get('temperature', .6) <= .6 else 'no', seen=seen)
    return seen


def test_default_grid_is_preregistered():
    assert list(DEFAULT_GRID) == ['default', 'temp-0.6', 'temp-0.8', 'presence-0.5', 'no-thinking', 'temp-0.6-no-thinking']
    assert DEFAULT_GRID == {'default': {}, 'temp-0.6': {'temperature': 0.6}, 'temp-0.8': {'temperature': 0.8},
                            'presence-0.5': {'presence_penalty': 0.5}, 'no-thinking': {'enable_thinking': False},
                            'temp-0.6-no-thinking': {'temperature': 0.6, 'enable_thinking': False}}
    assert grid_hash(DEFAULT_GRID) == '11ed4638f080ed85ba3c000fefe2623d72faceaad1ec0789ef9641674ff38405'


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
    assert not out.with_name('grid-results.partial').exists()


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


def test_non_default_cell_can_win(workspace, fake_runner):
    root, _, _ = workspace
    fake_runner(lambda g: 'yes' if g.get('temperature', .6) >= .8 else 'no')
    (root / 'grid.json').write_text(json.dumps({'hot': {'temperature': .8}}))
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json'), '--output-dir', str(root / 'g')])
    assert result.exit_code == 0, result.output
    summary = json.loads((root / 'g/grid-summary.json').read_text())
    assert summary['best_cell'] == 'hot' and summary['best_minus_default'] == 100 and summary['preregistered'] is False
    assert [c['name'] for c in summary['cells']] == ['default', 'hot']
    assert 'Best: hot 100.0 (+100.0 over default)' in result.output


@pytest.mark.parametrize('content', [[], {}, {'a b': {}}, {'x': 'x'}, {'x': {'temperature': '0.6'}},
                                     {'x': {'max_tokens': True}}, {'x': {'max_tokens': 0}}, {'x': {'temperature': -1}},
                                     {'x': {'enable_thinking': 1}}, {'default': {'temperature': .8}}, {'grid-summary': {}}])
def test_load_grid_rejects_bad_shapes(tmp_path, content):
    path = tmp_path / 'grid.json'
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_grid(path)


def test_unknown_cell_fails_before_generating(workspace, calls):
    root, _, _ = workspace
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--cells', 'nope', '--output-dir', str(root / 'g')])
    assert result.exit_code == 1 and 'Unknown cell(s)' in result.output
    assert not calls and not (root / 'g').exists() and not (root / 'g.partial').exists()


def test_tie_prefers_default_over_earlier_cell():
    cells = [{'name': 'hot', 'validation_score': 5}, {'name': 'default', 'validation_score': 5}]
    assert best_cell(cells, 'default') == 'default'
    assert best_cell([{'name': 'a', 'validation_score': 5}, {'name': 'b', 'validation_score': 5}, {'name': 'default', 'validation_score': 1}], 'default') == 'a'


def test_report_flags_overridden_cells_and_budget(workspace, calls):
    root, _, _ = workspace
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--output-dir', str(root / 'g')])
    assert result.exit_code == 0, result.output
    assert 'up to 480 output tokens' in result.output
    assert json.loads((root / 'g/default.json').read_text())['generation_overridden'] is False
    assert json.loads((root / 'g/temp-0.8.json').read_text())['generation_overridden'] is True
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--max-new-tokens', '4', '--output-dir', str(root / 'h')])
    assert result.exit_code == 0, result.output
    assert 'up to 96 output tokens' in result.output


def test_failed_run_leaves_partial_and_blocks_rerun(workspace, fake_runner):
    root, _, _ = workspace

    def answer(generation):
        if generation.get('temperature') == .8:
            raise RuntimeError('boom')
        return 'yes'
    fake_runner(answer)
    (root / 'grid.json').write_text(json.dumps({'hot': {'temperature': .8}}))
    args = ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json'), '--output-dir', str(root / 'g')]
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 1 and 'boom' in result.output
    assert (root / 'g.partial/default.json').exists() and not (root / 'g').exists()
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 1 and 'already exists' in result.output


@pytest.mark.parametrize('text', ['{"x": {"top_p": 1.5}}', '{"x": {"temperature": Infinity}}', '{"x": {"max_tokens": 40000}}'])
def test_invalid_values_fail_before_model_load(workspace, calls, text):
    root, _, _ = workspace
    (root / 'grid.json').write_text(text)
    out = root / 'g'
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json'), '--output-dir', str(out)])
    assert result.exit_code == 1
    assert not calls and not out.exists() and not (root / 'g.partial').exists()


def test_oversized_max_tokens_names_cell(workspace, calls):
    root, _, _ = workspace
    (root / 'grid.json').write_text('{"big": {"max_tokens": 40000}}')
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json')])
    assert "Cell 'big' sets max_tokens 40000, at or above max_model_len 34816." in result.output


@pytest.mark.parametrize('content', [{'.hidden': {}}, {'Hot': {'temperature': .8}, 'hot': {'temperature': .9}}])
def test_load_grid_rejects_bad_names(tmp_path, content):
    path = tmp_path / 'grid.json'
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_grid(path)


def test_registration_line_and_cells_requested(workspace, calls):
    root, _, _ = workspace
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--cells', 'temp-0.8', '--output-dir', str(root / 'g')])
    assert result.exit_code == 0, result.output
    assert f'Grid {grid_hash(DEFAULT_GRID)}, pre-registered default; cells: default, temp-0.8' in result.output
    assert json.loads((root / 'g/grid-summary.json').read_text())['cells_requested'] == ['default', 'temp-0.8']


def test_reserved_name_is_case_insensitive(workspace, calls):
    root, _, _ = workspace
    (root / 'grid.json').write_text(json.dumps({'Grid-Summary': {'temperature': .8}}))
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json'), '--output-dir', str(root / 'g')])
    assert result.exit_code == 1 and 'reserved' in result.output
    assert not calls


def test_huge_integer_fails_cleanly(workspace, calls):
    root, _, _ = workspace
    (root / 'grid.json').write_text('{"hot": {"temperature": 1' + '0' * 400 + '}}')
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json'), '--output-dir', str(root / 'g')])
    assert result.exit_code == 1 and "Generation key 'temperature'" in result.output
    assert not isinstance(result.exception, OverflowError) and not calls


def test_custom_baseline_name_is_reported(workspace, fake_runner):
    root, _, _ = workspace
    fake_runner(lambda g: 'yes' if g.get('temperature', .6) >= .8 else 'no')
    (root / 'grid.json').write_text(json.dumps({'base': {}, 'hot': {'temperature': .8}}))
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--grid', str(root / 'grid.json'), '--output-dir', str(root / 'g')])
    assert result.exit_code == 0, result.output
    assert json.loads((root / 'g/grid-summary.json').read_text())['default_cell'] == 'base'
    assert 'Best: hot 100.0 (+100.0 over base)' in result.output


def test_custom_grid_samples_preserve_shared_report_contract(workspace, calls):
    root, _, config = workspace
    config['protocol_version'] = 'delta-public-v1'
    config['generation']['public_validation'] = {'max_tokens': 20, 'temperature': .3, 'enable_thinking': False}
    (root / 'evaluation.json').write_text(json.dumps(config))
    (root / 'validation.json').write_text('{}')
    custom = root / 'custom.json'
    custom.write_text(json.dumps({'probe': [{'id': 'x', 'prompt': 'P', 'answer': 'yes'}]}))
    out = root / 'custom-grid'
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root), '--validation', str(custom),
        '--samples', '3', '--cells', 'temp-0.8', '--max-new-tokens', '4', '--output-dir', str(out)])
    assert result.exit_code == 0, result.output
    from delta_cli.artifacts import file_hash
    from delta_cli.compare import compare_reports
    summary = json.loads((out / 'grid-summary.json').read_text())
    assert summary['validation_sha256'] == file_hash(custom) and summary['samples_override'] == 3
    for name in ['default', 'temp-0.8']:
        report = json.loads((out / (name + '.json')).read_text())
        rows = [json.loads(line) for line in (out / (name + '.generations.jsonl')).read_text().splitlines()]
        assert [r['seed'] for r in rows] == [1000, 1001, 1002]
        assert report['evaluator'] == 'delta_cli.public_eval' and report['text_engine'] == 'lm-eval'
        assert report['generation']['temperature'] == (.3 if name == 'default' else .8)
        assert report['samples_override'] == 3 and not report['full_validation_set']
        assert report['validation_sha256'] == file_hash(custom)
        assert summary['effective_generation'][name] == report['generation']
    compared = compare_reports(out / 'default.json', out / 'temp-0.8.json', resamples=10)
    assert compared['generation_differs']


def test_pinned_grid_rejected_before_model_loading(workspace, monkeypatch):
    root, _, config = workspace
    config['protocol_version'] = 'delta-public-v1'
    (root / 'evaluation.json').write_text(json.dumps(config))
    from delta_cli import local
    monkeypatch.setattr(local, 'Runner', lambda *args: pytest.fail('Pinned suite must not load local model'))
    result = CliRunner().invoke(main.app, ['grid', '--workspace', str(root)])
    assert result.exit_code == 1 and 'explicit --validation file' in result.output
    assert 'inventory mismatch' not in result.output


def test_implicit_baseline_cannot_collide_with_case_variant(tmp_path):
    path = tmp_path / 'grid.json'
    path.write_text(json.dumps({'DEFAULT': {'temperature': .8}}))
    with pytest.raises(ValueError, match='implicit default'):
        load_grid(path)
