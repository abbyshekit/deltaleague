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
    queued = row("queued one", 50.0, 1.0, rank=2, validation=None, state="queued")
    serve(monkeypatch, [row("a", 50.0, 1.0), queued])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    assert "All rows are separable" in result.output
    assert "Not separable" not in result.output
    line = next(l for l in result.output.splitlines() if "queued one" in l)
    assert "—" in line and "50.00 ± 1.00" in line


def test_default_output_is_json(monkeypatch):
    serve(monkeypatch, DAY_ONE)
    result = runner.invoke(main.app, ARGS)
    assert json.loads(result.output) == {"config_id": "runtime-v5", "rows": DAY_ONE}


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
    assert "Not separable on the private test: a, b, c" in out
    assert "Private test standard errors from 0.10 to 10.00 points; two rows tie when their gap is within 1.96·√(se_a² + se_b²)." in out
    assert "up to" not in out
    assert not main.tied(scored[1], scored[2])
    assert names(main.tie_groups([scored[1], scored[2]])) == []


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


def test_tie_groups_bridge_and_order():
    bridged = [("a", 0.0, 1.0), ("b", 10.0, 1.0), ("c", 20.0, 1.0), ("d", 30.0, 1.0), ("e", 5.0, 3.0)]
    assert not main.tied(bridged[0], bridged[1])
    assert main.tied(bridged[4], bridged[0]) and main.tied(bridged[4], bridged[1])
    assert main.tied(bridged[1], bridged[2]) is False
    assert names(main.tie_groups(bridged[:2] + [bridged[4]])) == [["a", "b", "e"]]
    five = [("a", 0.0, 1.0), ("b", 4.0, 1.0), ("c", 8.0, 1.0), ("d", 12.0, 1.0), ("e", 6.0, 5.0)]
    assert names(main.tie_groups(five)) == [["a", "b", "c", "d", "e"]]
    disjoint = [("a", 0.0, 1.0), ("b", 100.0, 1.0), ("c", 101.0, 1.0), ("d", 1.0, 1.0)]
    assert names(main.tie_groups(disjoint)) == [["a", "d"], ["b", "c"]]


def test_empty_board(monkeypatch):
    serve(monkeypatch, [])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0].split() == ["rank", "name", "submitter", "validation", "test", "±", "se"]
    assert "No standard errors reported." in result.output


def test_bad_numbers_are_unscored(monkeypatch):
    serve(monkeypatch, [row("nan", 50.0, float("nan")), row("neg", 50.0, -1.0, rank=2),
                        row("inf", float("inf"), 1.0, rank=3), row("ok", 50.0, 1.0, rank=4)])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    out = result.output
    lines = {l.split()[1]: l for l in out.splitlines()[1:5]}
    assert all(v not in l.lower().replace(k, "") for k, l in lines.items() for v in ("nan", "inf", "-1"))
    assert "Not separable" not in out and "All rows are separable" in out
    assert lines["nan"].rstrip().endswith("50.00") and lines["neg"].rstrip().endswith("50.00")
    assert lines["inf"].rstrip().endswith("— ± 1.00")


def test_name_and_rank_fallbacks(monkeypatch):
    serve(monkeypatch, [row(None, 50.0, 1.0, rank=0), row("", 40.0, 1.0, rank=None)])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    first, second = result.output.splitlines()[1:3]
    assert first.split()[:2] == ["0", "—"] and second.split()[:2] == ["—", "—"]


def test_table_needs_model():
    result = runner.invoke(main.app, ["leaderboard", "--table"])
    assert result.exit_code != 0
    assert "--table needs --model: tie groups are per competition." in result.output


def test_table_sanitises_names(monkeypatch):
    serve(monkeypatch, [row("clean", 50.0, 1.0, submitter={"username": "u"})])
    clean = runner.invoke(main.app, ARGS + ["--table"]).output
    serve(monkeypatch, [row("a\x1b]0;pwn\x07\nFAKE 1 x", 50.0, 1.0, submitter={"username": "u\x1b[2J"})])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert not any(c in out for c in "\x1b\x07")
    assert out.count("\n") == clean.count("\n")
    assert "a?]0;pwn??FAKE 1 x" in out and "u?[2J" in out
    serve(monkeypatch, [row("n", 50.0, 1.0, submitter={"username": "x" * 60})])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert "x" * 24 in out and "x" * 25 not in out
