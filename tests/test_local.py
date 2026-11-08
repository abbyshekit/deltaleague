import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from click import unstyle
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from delta_cli import local, main
from delta_cli.artifacts import build_sparse, file_hash


def test_patch_build_rejects_a_modified_parent_and_an_existing_output(workspace):
    root, _, _ = workspace
    parent = root / 'model'
    modified = root / 'edited.safetensors'
    save_file({'layer.weight': torch.tensor([1., 9.])}, str(modified))
    config = parent / 'config.json'
    original = config.read_text()
    config.write_text('changed')
    output = root / 'built'
    args = ['patch', 'build', '--parent', str(parent), '--modified', str(modified), '--output', str(output)]
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code != 0
    assert 'changed' in result.output and 'Traceback' not in result.output
    assert not output.exists()
    config.write_text(original)
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 0, result.output
    assert (output / 'manifest.json').exists()
    again = CliRunner().invoke(main.app, args)
    assert again.exit_code != 0 and 'alreadyexists' in ''.join(unstyle(again.output).replace('│', '').split())
    previous = (output / 'manifest.json').read_bytes()
    (output / 'stale.txt').write_text('keep')
    save_file({'layer.weight': torch.tensor([1., 2.])}, str(modified))
    failed = CliRunner().invoke(main.app, args + ['--force'])
    assert failed.exit_code != 0 and 'Traceback' not in failed.output
    assert (output / 'manifest.json').read_bytes() == previous
    assert (output / 'stale.txt').read_text() == 'keep'
    save_file({'layer.weight': torch.tensor([1., 9.])}, str(modified))
    forced = CliRunner().invoke(main.app, args + ['--force'])
    assert forced.exit_code == 0, forced.output
    assert not (output / 'stale.txt').exists()


def test_parent_checkpoint_is_hashed_once(workspace, monkeypatch):
    root, metadata, _ = workspace
    calls = []
    real = file_hash

    def wrapped(path):
        calls.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(local, 'file_hash', wrapped)
    monkeypatch.setattr('delta_cli.artifacts.file_hash', wrapped)
    with local.model_directory(root, metadata):
        pass
    assert calls.count('model.safetensors') == 1


def test_patch_overlay_reconstructs_without_mutating_parent(workspace):
    root, metadata, _ = workspace
    modified = root / 'edited.safetensors'
    save_file({'layer.weight': torch.tensor([1., 9.])}, str(modified))
    build_sparse(root / 'model/model.safetensors', modified, root / 'patch', '00000000-0000-0000-0000-000000000002', 'a' * 64, 'tiny')
    original = file_hash(root / 'model/model.safetensors')
    with local.model_directory(root, metadata, patch=root / 'patch') as (directory, identity):
        assert torch.equal(load_file(str(directory / 'model.safetensors'))['layer.weight'], torch.tensor([1., 9.]))
        assert identity['kind'] == 'patch'
        assert (directory / 'config.json').read_text() == '{}'
    assert not directory.exists()
    assert file_hash(root / 'model/model.safetensors') == original
    manifest = root / 'patch/manifest.json'
    data = json.loads(manifest.read_text())
    data['parent_version_id'] = '00000000-0000-0000-0000-000000000003'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='parent'):
        with local.model_directory(root, metadata, patch=root / 'patch'):
            pass


def test_public_scoring_averages_samples_then_problems_then_benchmarks(workspace):
    root, _, config = workspace
    tasks, names, full, _ = local.public_examples(root, config)
    assert [t['seed'] for t in tasks] == [1000, 1001, 1000, 1000]
    answers = iter(['yes', 'no', 'yes', 'yes'])
    runner = SimpleNamespace(generate=lambda *a: {'text': next(answers), 'tokens': 3, 'generation_seconds': 1, 'truncated': False})
    stream = io.StringIO()
    report = local.evaluate(runner, tasks, config['generation'], stream, lambda *args: None)
    assert full and names == ['a', 'b']
    assert report['validation_benchmarks'] == {'a': 50, 'b': 100}
    assert report['validation_score'] == 75
    assert len(stream.getvalue().splitlines()) == 4
    assert local.public_examples(root, config, limit=1)[2] is False
    (root / 'validation.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        local.public_examples(root, config)


def test_thinking_and_choice_scoring():
    from delta_cli.public_eval.generation import final_answer
    assert local.score_answer({'answer': 'yes'}, final_answer('wrong reasoning</think> yes')) == 1
    assert local.score_answer({'answer': 'B', 'scorer': 'choice'}, r'Reasoning. \boxed{B}') == 1
    assert local.score_answer({'answer': '4', 'scorer': 'math'}, r'\boxed{4}') == 1


def test_ifeval_scorer_and_unknown_scorer(monkeypatch):
    class Instruction:
        def __init__(self, instruction_id):
            self.instruction_id = instruction_id
        def build_description(self, **kwargs):
            self.prompt = kwargs.get('prompt')
        def get_instruction_args(self):
            return {'prompt': None} if self.instruction_id == 'prompted' else None
        def check_following(self, answer):
            return answer == 'yes'

    from lm_eval.tasks.ifeval import instructions_registry
    monkeypatch.setattr(instructions_registry, 'INSTRUCTION_DICT', {'plain': Instruction, 'prompted': Instruction})
    plain = {'scorer': 'ifeval', 'prompt': 'Say yes', 'instruction_id_list': ['plain'], 'kwargs': [{}]}
    assert local.score_answer(plain, 'yes') == 1
    assert local.score_answer(plain, 'no') == 0
    prompted = {'scorer': 'ifeval', 'prompt': 'Say yes', 'instruction_id_list': ['prompted'], 'kwargs': [{}]}
    assert local.score_answer(prompted, 'yes') == 1
    with pytest.raises(ValueError, match='unknown_scorer'):
        local.score_answer({'scorer': 'bleu', 'answer': 'x'}, 'x')


def test_generate_rejects_an_over_budget_prompt_and_marks_truncation():
    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token = None
        pad_token = None
        def apply_chat_template(self, *args, **kwargs):
            return 'prompt'
        def __call__(self, text, return_tensors, add_special_tokens):
            return Batch(input_ids=torch.ones(1, 100, dtype=torch.long))
        def decode(self, tokens, skip_special_tokens=False):
            return 'answer'

    class Model:
        def generate(self, **kwargs):
            count = kwargs['max_new_tokens']
            return torch.zeros(1, 100 + count, dtype=torch.long)

    runner = local.Runner.__new__(local.Runner)
    runner.tokenizer = Tokenizer()
    runner.model = Model()
    runner.torch = torch
    runner.device = 'cpu'
    with pytest.raises(ValueError, match='max_model_len'):
        runner.generate('hi', {'max_tokens': 20, 'max_model_len': 110, 'temperature': 0}, 1)
    result = runner.generate('hi', {'max_tokens': 4, 'max_model_len': 200, 'temperature': 0, 'enable_thinking': False}, 1)
    assert result['truncated'] is True and result['tokens'] == 4 and result['text'] == 'answer'


def test_device_selection_and_unavailable_device():
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    assert local.select_device(fake, 'auto') == 'mps'
    assert local.select_device(fake, 'cpu') == 'cpu'
    with pytest.raises(ValueError, match='CUDA'):
        local.select_device(fake, 'cuda')


def test_cli_evaluation_report_and_existing_output_protection(workspace, monkeypatch):
    root, _, _ = workspace
    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, *args):
            pass
        def generate(self, *args):
            return {'text': 'yes', 'tokens': 2, 'generation_seconds': .1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', FakeRunner)
    runner = CliRunner()
    output = root / 'result.json'
    args = ['evaluate', '--workspace', str(root), '--limit', '1', '--max-new-tokens', '4', '--output', str(output)]
    result = runner.invoke(main.app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report['local_only'] and report['generation_overridden']
    assert report['full_validation_set'] is False
    assert report['validation_score'] == 100
    assert report['source_version_id'] == '00000000-0000-0000-0000-000000000002'
    original = output.read_bytes()
    result = runner.invoke(main.app, args)
    assert result.exit_code == 1 and 'already exists' in result.output
    assert output.read_bytes() == original


def test_weights_overlay_links_checkpoint_and_rejects_invalid_input(workspace):
    root, metadata, _ = workspace
    weights = root / 'candidate.safetensors'
    save_file({'layer.weight': torch.tensor([3., 4.])}, str(weights))
    with local.model_directory(root, metadata, weights=weights) as (directory, identity):
        link = directory / 'model.safetensors'
        assert identity['kind'] == 'weights' and link.is_symlink()
        assert file_hash(link) == file_hash(weights)
        assert (directory / 'config.json').is_symlink()
    with pytest.raises(ValueError, match='either'):
        with local.model_directory(root, metadata, weights=weights, patch=root / 'patch'):
            pass
    binary = root / 'weights.bin'
    binary.write_bytes(b'nope')
    with pytest.raises(ValueError, match='safetensors'):
        with local.model_directory(root, metadata, weights=binary):
            pass
    config = root / 'model' / 'config.json'
    original = config.read_text()
    config.write_text('mutated')
    try:
        with pytest.raises(ValueError, match='config.json'):
            with local.model_directory(root, metadata):
                pass
    finally:
        config.write_text(original)


def test_read_workspace_rejects_mismatched_config_model_and_parent(workspace):
    root, _, _ = workspace
    local.read_workspace(root)
    edits = [
        (root / 'delta-workspace.json', 'model_id', 'other'),
        (root / 'delta-workspace.json', 'config_id', 'other'),
        (root / 'delta-workspace.json', 'parent_version_id', '00000000-0000-0000-0000-000000000099'),
        (root / 'evaluation.json', 'model_id', 'other'),
        (root / 'model' / 'delta-model.json', 'model_id', 'other'),
    ]
    for path, key, value in edits:
        saved = path.read_text()
        data = json.loads(saved)
        data[key] = value
        path.write_text(json.dumps(data))
        try:
            with pytest.raises(ValueError, match='mismatch'):
                local.read_workspace(root)
        finally:
            path.write_text(saved)


def rehash(root, config, data):
    path = root / 'validation.json'
    path.write_text(json.dumps(data))
    updated = dict(config)
    updated['validation_sha256'] = file_hash(path)
    return updated


def test_public_examples_rejects_unknown_benchmark_empty_and_invalid_samples(workspace):
    root, _, config = workspace
    with pytest.raises(ValueError, match='Unknown benchmark'):
        local.public_examples(root, config, benchmark='missing')
    data = json.loads((root / 'validation.json').read_text())
    empty = rehash(root, config, {**data, 'a': []})
    with pytest.raises(ValueError, match='Empty benchmark'):
        local.public_examples(root, empty)
    bad_scorer = json.loads(json.dumps(data))
    bad_scorer['a'][0]['scorer'] = 'bleu'
    with pytest.raises(ValueError, match='Unsupported scorer'):
        local.public_examples(root, rehash(root, config, bad_scorer))
    for samples in (0, 17, '2'):
        counted = json.loads(json.dumps(data))
        counted['a'][0]['samples'] = samples
        with pytest.raises(ValueError, match='Invalid sample count'):
            local.public_examples(root, rehash(root, config, counted))
    with pytest.raises(ValueError, match='object of benchmarks'):
        local.public_examples(root, rehash(root, config, []))
    nested = {**data, 'a': {'prompt': 'A'}}
    with pytest.raises(ValueError, match='Invalid benchmark'):
        local.public_examples(root, rehash(root, config, nested))
    row = json.loads(json.dumps(data))
    row['a'] = ['yes']
    with pytest.raises(ValueError, match='Invalid example'):
        local.public_examples(root, rehash(root, config, row))


def test_leaderboard_version_model_mismatch_is_rejected(workspace, monkeypatch):
    root, _, _ = workspace
    monkeypatch.setattr(main, 'call', lambda *a, **kw: {'model_id': 'other'})
    monkeypatch.setattr(main, 'pull_model', lambda *a: pytest.fail('Should not download another model'))
    result = CliRunner().invoke(main.app, ['run', 'Hi', '--workspace', str(root), '--version', '00000000-0000-0000-0000-000000000001'])
    assert result.exit_code == 1 and 'another model' in result.output


def test_leaderboard_version_download_is_used_for_inference(workspace, monkeypatch):
    root, metadata, _ = workspace
    selected = '00000000-0000-0000-0000-000000000009'
    monkeypatch.setattr(main, 'call', lambda *a, **kw: {'model_id': 'tiny'})
    def pull(version_id, directory):
        assert version_id == selected
        directory.mkdir(parents=True)
        for artifact in metadata['files']:
            (directory / artifact['name']).write_bytes((root / 'model' / artifact['name']).read_bytes())
        return {**metadata, 'id': selected}
    monkeypatch.setattr(main, 'pull_model', pull)
    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, path, device):
            assert path == root / '.delta/models' / selected
        def generate(self, *args):
            return {'text': 'yes', 'tokens': 1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', FakeRunner)
    result = CliRunner().invoke(main.app, ['run', 'Hi', '--workspace', str(root), '--version', selected, '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['source_version_id'] == selected


@pytest.fixture
def fake_runner(monkeypatch):
    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, *args):
            pass
        def generate(self, *args):
            return {'text': 'yes', 'tokens': 2, 'generation_seconds': .1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', FakeRunner)


def test_load_validation(workspace):
    root, _, config = workspace
    path = root / 'custom.json'
    raw = json.dumps({'c': [{'id': 'c1', 'prompt': 'P', 'answer': 'yes'}]}).encode()
    path.write_bytes(raw)
    data, names, sha = local.load_validation(path)
    assert names == ['c'] and sha == hashlib.sha256(raw).hexdigest() and list(data) == ['c']
    tasks, _, full, _ = local.public_examples(root, config, validation_file=path)
    assert len(tasks) == 1 and full is False
    for content, message in [('{}', 'example objects'), ('{"b": ["text"]}', 'example objects'), ('nope', 'Cannot read'),
                             ('{"c": [{"id": "c1", "prompt": "P", "answer": "yes", "scorer": "nope"}]}', 'Unsupported scorer'),
                             ('{"c": [{"prompt": "P", "answer": "yes"}]}', 'need id and prompt')]:
        path.write_text(content)
        with pytest.raises(ValueError, match=message):
            local.public_examples(root, config, validation_file=path)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='changed'):
        local.load_validation(path, {**config, 'validation_names': ['c']})


def test_scorer_specific_fields(workspace, monkeypatch):
    root, _, config = workspace
    path = root / 'scorers.json'
    # lm_eval is optional; stub the dependency pre-import so ifeval validation runs without it.
    for name in ('lm_eval', 'lm_eval.tasks', 'lm_eval.tasks.ifeval', 'lm_eval.tasks.ifeval.instructions_registry'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, 'math_verify', ModuleType('math_verify'))

    def check(example):
        path.write_text(json.dumps({'c': [{'id': 'c1', 'prompt': 'P', **example}]}))
        return local.public_examples(root, config, validation_file=path)[0]

    ifeval = {'scorer': 'ifeval', 'instruction_id_list': ['length_constraints:number_words'], 'kwargs': [{'num_words': 5, 'relation': None}]}
    assert len(check(ifeval)) == 1  # no answer needed
    assert len(check({**ifeval, 'answer': 'ignored'})) == 1
    for bad, message in [({k: v for k, v in ifeval.items() if k != 'instruction_id_list'}, 'c1 .ifeval. needs instruction_id_list'),
                         ({**ifeval, 'instruction_id_list': []}, 'needs instruction_id_list'),
                         ({**ifeval, 'instruction_id_list': ['']}, 'needs instruction_id_list'),
                         ({**ifeval, 'kwargs': []}, 'c1 .ifeval. needs kwargs'),
                         ({**ifeval, 'kwargs': [{}, {}]}, 'needs kwargs'),
                         ({**ifeval, 'kwargs': [None]}, 'needs kwargs'),
                         ({k: v for k, v in ifeval.items() if k != 'kwargs'}, 'needs kwargs')]:
        with pytest.raises(ValueError, match=message):
            check(bad)
    for scorer in ('exact', 'choice', 'math'):
        for answer in ('x', 3, 2.5):
            assert len(check({'scorer': scorer, 'answer': answer})) == 1
        for bad in ({}, {'answer': None}, {'answer': ''}, {'answer': True}, {'answer': ['x']}):
            with pytest.raises(ValueError, match=f'c1 .{scorer}. needs answer'):
                check({'scorer': scorer, **bad})
    with pytest.raises(ValueError, match='c1 .exact. needs answer'):
        check({})  # default scorer is exact


def test_cli_custom_validation_file(workspace, fake_runner):
    root, _, config = workspace
    path = root / 'custom.json'
    path.write_text(json.dumps({'c': [{'id': f'c{i}', 'prompt': 'P', 'answer': 'yes'} for i in range(2)],
                                'd': [{'id': 'd1', 'prompt': 'P', 'answer': 'yes'}]}))
    runner = CliRunner()
    args = ['evaluate', '--workspace', str(root), '--max-new-tokens', '4']
    default = root / 'default.json'
    result = runner.invoke(main.app, args + ['--output', str(default)])
    assert result.exit_code == 0, result.output
    assert json.loads(default.read_text())['validation_file'].endswith('validation.json')
    (root / 'validation.json').write_text((root / 'validation.json').read_text() + ' ')
    assert runner.invoke(main.app, args + ['--output', str(root / 'x.json')]).exit_code == 1
    output = root / 'custom-result.json'
    result = runner.invoke(main.app, args + ['--validation', str(path), '--benchmark', 'c', '--limit', '1', '--output', str(output)])
    assert result.exit_code == 0, result.output
    assert '1 generations' in result.output
    report = json.loads(output.read_text())
    assert report['validation_file'] == str(path.resolve())
    assert report['validation_sha256'] != config['validation_sha256']
    assert report['validation_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report['full_validation_set'] is False
    assert report['validation_benchmarks'] == {'c': 100}


def test_load_validation_error_paths(workspace):
    root, _, config = workspace
    with pytest.raises(ValueError, match='Cannot read'):
        local.load_validation(root / 'missing.json')
    mixed = root / 'mixed.json'
    mixed.write_text('{"c": [{"id": "x", "prompt": "p", "answer": "a", "scorer": "exact"}, "text"]}')
    with pytest.raises(ValueError, match='example objects'):
        local.load_validation(mixed)
    with pytest.raises(ValueError, match='inventory'):
        local.load_validation(root / 'validation.json', {**config, 'validation_names': ['c']})


def test_patch_build_requires_checkpoint_in_inventory(workspace, monkeypatch):
    root, _, _ = workspace
    parent = root / 'model'
    path = parent / 'delta-model.json'
    metadata = json.loads(path.read_text())
    metadata['files'] = []
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(main, 'build_sparse', lambda *args: pytest.fail('Unverified checkpoint used'))
    result = CliRunner().invoke(main.app, ['patch', 'build', '--parent', str(parent), '--modified', str(root / 'edited.safetensors'), '--output', str(root / 'built')])
    assert result.exit_code != 0
    assert 'inventory' in result.output
    assert not (root / 'built').exists()


@pytest.mark.parametrize('fail_publish', [False, True])
def test_publish_preserves_unrelated_backup_and_rolls_back(tmp_path, monkeypatch, fail_publish):
    output, staging, unrelated = (tmp_path / name for name in ['submission', 'staging', '.submission.replaced'])
    for directory in (output, staging, unrelated):
        directory.mkdir()
    (output / 'old').write_text('old')
    (staging / 'new').write_text('new')
    (unrelated / 'keep').write_text('keep')
    rename = Path.rename
    def failing_rename(path, target):
        if path == staging:
            raise OSError('cannot publish')
        return rename(path, target)
    if fail_publish:
        monkeypatch.setattr(Path, 'rename', failing_rename)
        with pytest.raises(OSError, match='cannot publish'):
            main.publish_directory(staging, output)
        assert (output / 'old').read_text() == 'old'
    else:
        main.publish_directory(staging, output)
        assert (output / 'new').read_text() == 'new'
    assert (unrelated / 'keep').read_text() == 'keep'
    assert not list(tmp_path.glob('.submission.backup-*'))


def test_custom_validation_routes_before_pinned_protocol(workspace, fake_runner, monkeypatch):
    root, _, config = workspace
    config['protocol_version'] = 'delta-public-v1'
    (root / 'evaluation.json').write_text(json.dumps(config))
    (root / 'validation.json').write_text('{"not": "a legacy suite"}')
    path = root / 'custom.json'
    path.write_text(json.dumps({'custom': [{'id': 'x', 'prompt': 'P', 'answer': 'yes'}]}))
    from delta_cli.public_eval import cli
    monkeypatch.setattr(cli, 'run_command', lambda **kwargs: pytest.fail('Custom data sent to pinned evaluator'))
    output = root / 'custom-result.json'
    args = ['evaluate', '--workspace', str(root), '--validation', str(path), '--output', str(output)]
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report['validation_kind'] == 'custom' and report['full_validation_set'] is False
    assert report['validation_sha256'] == file_hash(path)
    rejected = CliRunner().invoke(main.app, args + ['--endpoint', 'http://localhost:8000/v1'])
    assert rejected.exit_code == 1 and 'do not combine' in rejected.output


def test_custom_validation_rejects_terminal_control_in_benchmark(workspace):
    root, _, config = workspace
    path = root / 'custom.json'
    path.write_text(json.dumps({'unsafe\x1b[2J': [{'id': 'x', 'prompt': 'p', 'answer': 'a'}]}))
    with pytest.raises(ValueError, match='Invalid benchmark name'):
        local.public_examples(root, config, validation_file=path)


def test_cli_samples_bounds(workspace, monkeypatch):
    root, _, _ = workspace
    calls = []

    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, *args):
            pass
        def generate(self, *args):
            calls.append(args)
            return {'text': 'yes', 'tokens': 2, 'generation_seconds': .1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', FakeRunner)
    runner = CliRunner()
    for bad in ('0', '17'):
        result = runner.invoke(main.app, ['evaluate', '--workspace', str(root), '--samples', bad,
                                          '--max-new-tokens', '4', '--output', str(root / f'bad{bad}.json')])
        assert result.exit_code != 0
        assert not calls
        assert not (root / f'bad{bad}.json').exists()
    output = root / 'sixteen.json'
    result = runner.invoke(main.app, ['evaluate', '--workspace', str(root), '--samples', '16',
                                      '--max-new-tokens', '4', '--output', str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())['samples_override'] == 16
    assert len(output.with_suffix('.generations.jsonl').read_text().splitlines()) == 48


def test_custom_sample_override_tracks_hash_and_seeds(workspace, fake_runner):
    root, _, config = workspace
    path = root / 'custom.json'
    path.write_text(json.dumps({'custom': [{'id': 'x', 'prompt': 'P', 'answer': 'yes', 'samples': 1}]}))
    tasks, names, full, sha = local.public_examples(root, config, validation_file=path, samples=3)
    assert [t['seed'] for t in tasks] == [1000, 1001, 1002]
    assert names == ['custom'] and not full and sha == file_hash(path)
    config['protocol_version'] = 'delta-public-v1'
    (root / 'evaluation.json').write_text(json.dumps(config))
    output = root / 'sampled.json'
    result = CliRunner().invoke(main.app, ['evaluate', '--workspace', str(root), '--validation', str(path),
                                          '--samples', '3', '--output', str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report['samples_override'] == 3 and not report['full_validation_set']
    assert report['validation_sha256'] == sha
    assert len(local.generations_path(output).read_text().splitlines()) == 3


def test_pinned_suite_rejects_samples_before_dispatch(workspace, monkeypatch):
    root, _, config = workspace
    config['protocol_version'] = 'delta-public-v1'
    (root / 'evaluation.json').write_text(json.dumps(config))
    from delta_cli.public_eval import cli
    monkeypatch.setattr(cli, 'run_command', lambda **kwargs: pytest.fail('Ignored sample override'))
    result = CliRunner().invoke(main.app, ['evaluate', '--workspace', str(root), '--samples', '2'])
    assert result.exit_code == 1 and '--samples is only supported' in result.output
