"""Exercise the real shared text engine with offline inference fixtures."""
import io
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from delta_cli import local
from delta_cli.public_eval import runner, text


def backend(answer, seen=None):
    def generate(prompt, generation, seed):
        if seen is not None:
            seen.append((prompt, generation, seed))
        return {'text': answer, 'tokens': 2, 'truncated': False, 'generation_seconds': .01}
    return SimpleNamespace(generate=generate)


def test_pinned_and_custom_text_use_the_same_engine(workspace, tmp_path, monkeypatch):
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from delta_cli.public_eval.common import digest
    root, _, config = workspace
    tasks, _, _, _ = local.public_examples(root, config)
    calls = []
    real = evaluator.simple_evaluate

    def traced(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)
    monkeypatch.setattr(evaluator, 'simple_evaluate', traced)
    custom = runner.evaluate_custom(runner=backend('yes'), tasks=tasks, generation=config['generation'],
                                   stream=io.StringIO(), progress=lambda *args: None)
    task = calls[0]['tasks'][0]

    class ServedBackend(LM):
        tokenizer_name = 'fixture'
        def __init__(self, **kwargs):
            super().__init__()
            self.tokenizer = SimpleNamespace(apply_chat_template=lambda history, **kw: history[-1]['content'])
        def generate_until(self, requests):
            return ['yes' for _ in requests]
        def loglikelihood(self, requests):
            raise AssertionError('Unexpected likelihood request')
        def loglikelihood_rolling(self, requests):
            raise AssertionError('Unexpected likelihood request')
        def chat_template(self, *args):
            return 'fixture'

    api = ModuleType('lm_eval.models.openai_completions')
    api.LocalCompletionsAPI = ServedBackend
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setattr(text, 'load_tasks', lambda *args: {'delta_custom': task})
    selection = {'output_type': 'generate_until', 'num_fewshot': 0, 'population': len(tasks),
                 'indices': list(range(len(tasks))), 'doc_sha256': [digest(t) for t in tasks]}
    manifest = {'seed': 1000, 'sha256': 'fixture', 'dataset_revisions': {},
                'text': {'fixture': {'task': 'delta_custom', 'selection': {'delta_custom': selection}}}}
    pinned = text.run(manifest, 'fixture', 'http://unused/v1', 'fixture', 'unused', tmp_path / 'native.json')
    assert len(calls) == 2
    assert custom['validation_score'] == 100
    assert all(sample['score'] == 1 for sample in pinned['samples']['delta_custom'])
    assert custom['evaluator'] == 'delta_cli.public_eval' and custom['text_engine_version'] == '0.4.13'


@pytest.mark.parametrize('answer,score', [('unfinished reasoning', 0), ('reasoning</think>yes', 100)])
def test_custom_thinking_uses_shared_final_answer(workspace, answer, score):
    root, _, config = workspace
    tasks, _, _, _ = local.public_examples(root, config)
    stream = io.StringIO()
    result = runner.evaluate_custom(runner=backend(answer), tasks=tasks,
        generation={**config['generation'], 'enable_thinking': True}, stream=stream, progress=lambda *args: None)
    assert result['validation_score'] == score
    assert json.loads(stream.getvalue().splitlines()[0])['text'] == answer


def test_custom_ifeval_uses_native_metric(workspace):
    from lm_eval.tasks.ifeval.utils import process_results
    root, _, config = workspace
    example = {'id': 'x', 'prompt': 'Say yes in lowercase', 'scorer': 'ifeval',
               'instruction_id_list': ['change_case:english_lowercase'], 'kwargs': [{}]}
    path = root / 'custom.json'
    path.write_text(json.dumps({'ifeval': [example]}))
    tasks, _, _, _ = local.public_examples(root, config, validation_file=path)
    answer = 'this is a complete english sentence with all lowercase words.'
    result = runner.evaluate_custom(runner=backend(answer), tasks=tasks, generation=config['generation'],
                                   stream=io.StringIO(), progress=lambda *args: None)
    native = process_results({**example, 'key': 'x'}, [answer])['prompt_level_strict_acc']
    assert result['validation_score'] == 100 * native == 100


def test_custom_missing_coverage_is_not_a_complete_report(workspace, monkeypatch):
    root, _, config = workspace
    tasks, _, _, _ = local.public_examples(root, config)
    monkeypatch.setattr(text, 'evaluate_tasks', lambda **kw: {'samples': {'delta_custom': []}})
    stream = io.StringIO()
    with pytest.raises(ValueError, match='coverage'):
        runner.evaluate_custom(runner=backend('yes'), tasks=tasks, generation=config['generation'],
                               stream=stream, progress=lambda *args: None)
    assert stream.getvalue() == ''


def test_custom_inherits_public_generation_profile(workspace, monkeypatch):
    from typer.testing import CliRunner
    from delta_cli import main
    root, _, config = workspace
    config.update(protocol_version='delta-public-v1', generation={'max_tokens': 999,
        'public_validation': {'max_tokens': 4, 'temperature': .8, 'enable_thinking': False}})
    (root / 'evaluation.json').write_text(json.dumps(config))
    seen = []
    fake = backend('yes', seen)
    fake.details = {'device': 'cpu'}
    monkeypatch.setattr(local, 'Runner', lambda *args: fake)
    output = root / 'report.json'
    result = CliRunner().invoke(main.app, ['evaluate', '--workspace', str(root), '--validation',
        str(root / 'validation.json'), '--output', str(output)])
    assert result.exit_code == 0, result.output
    assert seen and all(g['max_tokens'] == 4 and g['temperature'] == .8 for _, g, _ in seen)
    assert json.loads(output.read_text())['generation'] == config['generation']['public_validation']
