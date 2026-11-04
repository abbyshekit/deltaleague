import json

from typer.testing import CliRunner

from delta_cli import main

runner = CliRunner()
ARGS = ["leaderboard", "--model", "qwen35-08b-v1"]


def row(name, test, se, rank=1, validation=40.0, state="complete", submitter=None, parent_id="p"):
    return {"name": name, "submitter": submitter, "rank": rank, "parent_id": parent_id,
            "result": {"state": state, "validation_score": validation, "test_score": test,
                       "stats": {"test_standard_error": se}}}


def serve(monkeypatch, rows):
    monkeypatch.setattr(main, "call", lambda *a, **k: {"config_id": "runtime-v5", "rows": rows})


DAY_ONE = [
    row("Original model", 33.33333333333333, 16.666666666666664, validation=40.0, parent_id=None),
    row("Published weight-space edit reproduction", 33.33333333333333, 16.666666666666664, validation=46.666666666666664,
        submitter={"username": "ilblackdragon"}),
]


def test_table_day_one_ties(monkeypatch):
    serve(monkeypatch, DAY_ONE)
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert "Original model" in out and "Published weight-space edit reproduction" in out
    assert "33.33 ± 16.67" in out and "original" in out and "ilblackdragon" in out
    assert "Not separable on the private test: Original model, Published weight-space edit reproduction" in out
    assert "a difference under 46.20 is noise" in out


def test_table_separable(monkeypatch):
    serve(monkeypatch, [row("a", 50.0, 1.0), row("b", 40.0, 1.0, rank=2)])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert "All rows are separable on the private test." in out
    assert "Not separable" not in out


def test_table_incomplete_row(monkeypatch):
    queued = row("queued one", None, None, rank=2, validation=None, state="queued")
    serve(monkeypatch, [row("a", 50.0, 1.0), queued])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    assert "—" in result.output
    assert "All rows are separable" in result.output


def test_default_output_is_json(monkeypatch):
    serve(monkeypatch, DAY_ONE)
    result = runner.invoke(main.app, ARGS)
    assert len(json.loads(result.output)["rows"]) == 2


def names(groups):
    return [[r[0] for r in g] for g in groups]


def test_tie_groups_chain(monkeypatch):
    scored = [("a", 0.0, 1.0), ("b", 2.0, 1.0), ("c", 4.0, 1.0)]
    assert not main.tied(scored[0], scored[2])
    assert names(main.tie_groups(scored)) == [["a", "b", "c"]]
    serve(monkeypatch, [row("a", 0.0, 1.0), row("b", 2.0, 1.0, rank=2), row("c", 4.0, 1.0, rank=3)])
    assert "Not separable on the private test: a, b, c" in runner.invoke(main.app, ARGS + ["--table"]).output


def test_tie_groups_uneven_se(monkeypatch):
    scored = [("a", 0.0, 10.0), ("b", 1.0, 0.1), ("c", 5.0, 0.1)]
    assert names(main.tie_groups(scored)) == [["a", "b", "c"]]
    serve(monkeypatch, [row("a", 0.0, 10.0), row("b", 1.0, 0.1, rank=2), row("c", 5.0, 0.1, rank=3)])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert "Not separable on the private test: a, b, c" in out and "up to 10.00" in out


def test_identical_scores_with_zero_se_are_tied(monkeypatch):
    serve(monkeypatch, [row("a", 100.0, 0.0), row("b", 100.0, 0.0, rank=2)])
    assert "Not separable" in runner.invoke(main.app, ARGS + ["--table"]).output


def test_null_username_and_parent_rows(monkeypatch):
    serve(monkeypatch, [row("orig", 50.0, 1.0, parent_id=None),
                        row("anon", 40.0, 1.0, rank=2, submitter={"username": None}, parent_id="x")])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    lines = {l.split()[1]: l for l in result.output.splitlines()[1:3]}
    assert "original" in lines["orig"] and "—" in lines["anon"] and "original" not in lines["anon"]
