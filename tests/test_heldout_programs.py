"""Generated-program checks use the shared WASI boundary, never a host subprocess."""
import importlib.util
import os
from pathlib import Path

import pytest
from delta_cli.public_eval import wasi

source = Path(__file__).parents[1] / 'benchmarks/heldout-v1/program_check.py'
spec = importlib.util.spec_from_file_location('heldout_program_check', source)
program_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program_check)


def test_missing_sandbox_fails_closed(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('python.wasm')
    monkeypatch.setattr(wasi, 'sandbox', unavailable)
    with pytest.raises(FileNotFoundError):
        program_check.run_program('def solve(): return 73', 73)


def test_failed_preflight_stops_build(monkeypatch):
    monkeypatch.setattr(wasi, 'sandbox', lambda source: {'passed': False, 'reason': 'test_failure'})
    with pytest.raises(RuntimeError, match='preflight'):
        program_check.run_program('def solve(): return 73', 73)


def test_program_failure_is_an_item_veto(monkeypatch):
    results = iter([{'passed': True}, {'passed': False, 'reason': 'resource_limit'}])
    monkeypatch.setattr(wasi, 'sandbox', lambda source: next(results))
    assert program_check.run_program('def solve(): return 73', 73) == 'program failed sandbox validation: resource_limit'


native = pytest.mark.skipif(os.getenv('DELTA_TEST_HELDOUT_WASI') != '1',
                            reason='Requires the pinned WASI interpreter and Wasmtime')


@native
def test_native_correct_and_incorrect_answers():
    assert program_check.run_program('def solve(): return 73', 73) is None
    assert program_check.run_program('def solve(): return 74', 73) is not None
    assert program_check.run_program('def solve(): return float("nan")', 73) is not None
    assert program_check.run_program('def solve(): return float("inf")', 73) is not None


@native
def test_native_aliases_cannot_read_or_write_host_files(tmp_path):
    marker = tmp_path / 'review-marker'
    marker.write_text('73')
    read = f'def solve():\n    reader = open\n    return float(reader({str(marker)!r}).read())'
    write = f'def solve():\n    writer = open\n    writer({str(marker)!r}, "w").write("changed")\n    return 73'
    for program in (read, write):
        assert program_check.veto_program(program) is None  # Defeats the old screen.
        assert program_check.run_program(program, 73) is not None
    assert marker.read_text() == '73'
