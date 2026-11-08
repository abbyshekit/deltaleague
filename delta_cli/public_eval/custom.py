"""In-memory custom JSON tasks for the shared native text evaluation engine."""
import json
from importlib.metadata import version
import statistics

from . import text
from .custom_metrics import score_answer
from .generation import final_answer


def check_runtime():
    if version('lm-eval') != '0.4.13':
        raise ValueError('Custom evaluation requires lm-eval==0.4.13, matching the public evaluator.')


def run(runner, tasks, generation, stream, progress):
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task
    from lm_eval.config.task import TaskConfig
    from lm_eval.tasks import TaskManager

    check_runtime()
    from langdetect import DetectorFactory
    DetectorFactory.seed = 1000
    if not tasks:
        raise ValueError('Custom evaluation needs at least one example.')

    class CustomTask(Task):
        VERSION = 1
        OUTPUT_TYPE = 'generate_until'

        def download(self, *args, **kwargs):
            pass  # The caller has already loaded and validated the exact input bytes.

        def has_training_docs(self):
            return False

        def has_validation_docs(self):
            return False

        def has_test_docs(self):
            return True

        def test_docs(self):
            return tasks

        def doc_to_text(self, doc):
            return doc['example']['prompt']

        def doc_to_target(self, doc):
            return str(doc['example'].get('answer', ''))

        def construct_requests(self, doc, ctx, **kwargs):
            return Instance(request_type='generate_until', doc=doc, arguments=(ctx, generation),
                            idx=0, metadata=kwargs['metadata'])

        def process_results(self, doc, results):
            return {'score': score_answer(doc['example'], results[0])}

        def aggregation(self):
            return {'score': statistics.mean}

        def higher_is_better(self):
            return {'score': True}

    class LocalBackend(LM):
        """Inference-only adapter; request construction and scoring belong to lm-eval."""
        def __init__(self):
            super().__init__()
            self.outputs = {}

        def loglikelihood(self, requests):
            raise ValueError('Custom JSON tasks require text generation.')

        def loglikelihood_rolling(self, requests):
            raise ValueError('Custom JSON tasks require text generation.')

        def generate_until(self, requests):
            answers = []
            for request in requests:
                result = runner.generate(request.args[0], generation, request.doc['seed'])
                self.outputs[request.doc_id] = result
                answer = result['text']
                answers.append(final_answer(answer) if generation.get('enable_thinking', False) else answer)
                progress(len(self.outputs), len(tasks), request.doc['benchmark'])
            return answers

    task = CustomTask()
    task._config = TaskConfig(task='delta_custom', output_type='generate_until', num_fewshot=0,
                              generation_kwargs={'until': [], 'max_gen_toks': generation.get('max_tokens', 32768)})
    backend = LocalBackend()
    report = text.evaluate_tasks(model=backend, tasks=[task], task_manager=TaskManager(include_defaults=False),
        bootstrap_iters=0, log_samples=True,
        apply_chat_template=False, random_seed=1000, numpy_random_seed=1000, torch_random_seed=1000,
        fewshot_random_seed=1000)
    samples = report['samples']['delta_custom']
    by_id = {sample['doc_id']: sample for sample in samples if sample['filter'] == 'none'}
    if len(samples) != len(tasks) or set(by_id) != set(range(len(tasks))) or set(backend.outputs) != set(by_id):
        raise ValueError('Incomplete or unexpected custom evaluation coverage.')
    rows = []
    for index, task in enumerate(tasks):
        row = {k: v for k, v in task.items() if k != 'example'}
        row.update(example_id=task['example']['id'], score=float(by_id[index]['score']), **backend.outputs[index])
        stream.write(json.dumps(row) + '\n')
        rows.append(row)
    stream.flush()
    # Keep per-example weighting when different examples request different repeats.
    grouped = {}
    for row in rows:
        grouped.setdefault(row['benchmark'], {}).setdefault(row['example_index'], []).append(row['score'])
    components = {name: statistics.mean(statistics.mean(scores) for scores in examples.values()) * 100
                  for name, examples in grouped.items()}
    return {'validation_score': statistics.mean(components.values()), 'validation_benchmarks': components,
            'output_tokens': sum(r['tokens'] for r in rows),
            'generation_seconds': sum(r['generation_seconds'] for r in rows),
            'truncation_rate': statistics.mean(r['truncated'] for r in rows),
            'evaluator': 'delta_cli.public_eval', 'text_engine': 'lm-eval', 'text_engine_version': version('lm-eval'),
            'task_protocol': 'custom-json-v1'}
