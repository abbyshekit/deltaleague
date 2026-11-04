from contextlib import nullcontext
import hashlib
import json

import httpx
import pytest
from typer.testing import CliRunner

from delta_cli import main

runner = CliRunner()


def artifact(data=b'weights'):
    return {"name": "model.safetensors", "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "url": "https://blob.invalid/?secret=hidden"}


def test_download_integrity_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / 'model.safetensors'
    path.write_bytes(b'original')
    monkeypatch.setattr(main.httpx, 'stream', lambda *a, **k: nullcontext(httpx.Response(200, content=b'bad', request=httpx.Request('GET', a[1]))))
    with pytest.raises(main.typer.Exit):
        main.download_artifact(artifact(), tmp_path, force=True)
    assert path.read_bytes() == b'original'
    assert not list(tmp_path.glob('*.part'))


def test_download_reuses_only_verified_files(tmp_path, monkeypatch):
    path = tmp_path / 'model.safetensors'
    path.write_bytes(b'weights')
    monkeypatch.setattr(main.httpx, 'stream', lambda *a, **k: pytest.fail('Unnecessary network request'))
    main.download_artifact(artifact(), tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(main.typer.BadParameter, match='different contents'):
        main.download_artifact(artifact(), tmp_path)


def test_init_records_reproducible_workspace_without_signed_urls(tmp_path, monkeypatch):
    validation = b'{"examples": []}'
    config = {"id": "config-v1", "validation_names": ["math"],
              "validation_download": "https://blob.invalid/validation?secret=hidden",
              "validation_sha256": hashlib.sha256(validation).hexdigest(), "validation_size_bytes": len(validation)}
    def call(method, path, **kwargs):
        if path == 'leaderboard':
            return {"config_id": "config-v1", "rows": [{"version_id": "parent", "parent_id": None}]}
        if path.startswith('evaluation-configs/'):
            return config
        return {"id": "parent", "model_id": "model", "content_hash": "hash", "files": [artifact()]}
    monkeypatch.setattr(main, 'call', call)
    monkeypatch.setattr(main.httpx, 'stream', lambda *a, **k: nullcontext(httpx.Response(200, content=validation if '/validation' in a[1] else b'weights', request=httpx.Request('GET', a[1]))))
    result = runner.invoke(main.app, ['competition', 'init', 'model', '--output', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / 'delta-workspace.json').read_text())['parent_version_id'] == 'parent'
    assert (tmp_path / 'validation.json').read_bytes() == validation
    for path in tmp_path.rglob('*.json'):
        assert 'secret=hidden' not in path.read_text()
    # A different model cannot silently replace the workspace's parent.
    result = runner.invoke(main.app, ['competition', 'init', 'other', '--parent', 'parent', '--output', str(tmp_path)])
    assert result.exit_code != 0
    assert 'different model' in result.output


def test_api_connection_failure_does_not_print_credentials(monkeypatch):
    def fail():
        raise httpx.ConnectError('https://user:secret@server.invalid')
    monkeypatch.setattr(main, 'client', fail)
    result = runner.invoke(main.app, ['competition', 'list'])
    assert result.exit_code == 1
    assert 'secret' not in result.output
    assert 'Could not reach' in result.output


def fake_api(monkeypatch, handler):
    monkeypatch.setattr(main, 'client', lambda: httpx.Client(base_url='https://x', transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(main.time, 'sleep', lambda *_: None)


def login_handler(failures):
    polls = []

    def handler(request):
        if request.url.path.endswith('/cli-auth'):
            return httpx.Response(200, json={'id': 'i', 'secret': 's', 'verification_url': 'https://x/approve', 'interval': 0})
        polls.append(request)
        if len(polls) <= len(failures):
            failure = failures[len(polls) - 1]
            if failure is None:
                raise httpx.ConnectError('down', request=request)
            return httpx.Response(failure)
        return httpx.Response(200, json={'token': 't'})
    return handler, polls


def test_login_poll_survives_transient_errors(tmp_path, monkeypatch):
    handler, polls = login_handler([None, None])
    fake_api(monkeypatch, handler)
    config = tmp_path / 'config' / 'config.json'
    monkeypatch.setattr(main, 'config_path', lambda: config)
    result = runner.invoke(main.app, ['login', '--no-browser'])
    assert result.exit_code == 0
    assert 'Signed in.' in result.output
    assert len(polls) == 3
    assert '"token": "t"' in config.read_text()


def test_login_poll_survives_transient_server_error(tmp_path, monkeypatch):
    handler, polls = login_handler([503])
    fake_api(monkeypatch, handler)
    config = tmp_path / 'config.json'
    monkeypatch.setattr(main, 'config_path', lambda: config)
    result = runner.invoke(main.app, ['login', '--no-browser'])
    assert result.exit_code == 0
    assert len(polls) == 2
    assert '"token": "t"' in config.read_text()


def test_call_retries_are_bounded(monkeypatch, capsys):
    attempts = []

    def handler(request):
        attempts.append(request)
        raise httpx.ConnectError('down', request=request)
    fake_api(monkeypatch, handler)
    with pytest.raises(main.typer.Exit):
        main.call('GET', 'x', retries=2)
    assert len(attempts) == 3
    assert 'Could not reach' in capsys.readouterr().err
    attempts.clear()
    with pytest.raises(main.typer.Exit):
        main.call('GET', 'x')
    assert len(attempts) == 1


def test_call_retries_429_then_succeeds(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(429) if len(attempts) == 1 else httpx.Response(200, json={'ok': 1})
    fake_api(monkeypatch, handler)
    assert main.call('GET', 'me', retries=2) == {'ok': 1}
    assert len(attempts) == 2


def test_call_stops_at_bound_on_persistent_5xx(monkeypatch, capsys):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, json=['down'])
    fake_api(monkeypatch, handler)
    with pytest.raises(main.typer.Exit):
        main.call('GET', 'me', retries=2)
    assert len(attempts) == 3
    assert 'API request failed (HTTP 503)' in capsys.readouterr().err


def test_call_does_not_retry_4xx(monkeypatch, capsys):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(404)
    fake_api(monkeypatch, handler)
    with pytest.raises(main.typer.Exit):
        main.call('GET', 'me', retries=5)
    assert len(attempts) == 1
    captured = capsys.readouterr()
    assert 'API request failed (HTTP 404)' in captured.out + captured.err


def test_login_poll_exhausts_and_exits(tmp_path, monkeypatch):
    polls = []

    def handler(request):
        if request.url.path.endswith('/cli-auth'):
            return httpx.Response(200, json={'id': 'i', 'secret': 's', 'verification_url': 'https://x/approve', 'interval': 0})
        polls.append(request)
        raise httpx.ConnectError('down', request=request)
    fake_api(monkeypatch, handler)
    monkeypatch.setattr(main, 'config_path', lambda: tmp_path / 'config.json')
    result = runner.invoke(main.app, ['login', '--no-browser'])
    assert result.exit_code == 1
    assert 'Could not reach' in result.output
    assert len(polls) == 6


def test_status_watch_survives_transient_errors(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= 2:
            raise httpx.ConnectError('down', request=request)
        return httpx.Response(200, json={'state': 'verified'})
    fake_api(monkeypatch, handler)
    result = runner.invoke(main.app, ['submission', 'status', 'x', '--watch'])
    assert result.exit_code == 0
    assert 'verified' in result.output
    calls.clear()

    def once(request):
        raise httpx.ConnectError('down', request=request)
    fake_api(monkeypatch, once)
    result = runner.invoke(main.app, ['submission', 'status', 'x'])
    assert result.exit_code == 1
    assert 'Could not reach' in result.output
