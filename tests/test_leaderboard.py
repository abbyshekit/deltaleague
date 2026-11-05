import json
import re

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


def cells(line):
    return re.split(r"\s{2,}", line.strip())


def gap_cell(out, name):
    line = next(l for l in out.splitlines() if name in l)
    return cells(line)[-1]


def test_table_day_one_equal_scores(monkeypatch):
    serve(monkeypatch, DAY_ONE)
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert "Original model" in out and "Published weight-space edit reproduction" in out
    assert "33.33 ± 16.67" in out and "original" in out and "ilblackdragon" in out
    assert gap_cell(out, "Original model") == "—"
    assert gap_cell(out, "Published weight-space edit reproduction") == "+0.00 *"
    assert "independence-based heuristic" in out and "not evidence two rows are equal" in out


def test_table_outside_band(monkeypatch):
    serve(monkeypatch, [row("a", 50.0, 1.0), row("b", 40.0, 1.0, rank=2)])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert gap_cell(out, "a  ") == "—" and gap_cell(out, "b  ") == "-10.00"
    assert out.count("*") == 1


def test_table_incomplete_row(monkeypatch):
    queued = row("queued one", 50.0, 1.0, rank=2, validation=None, state="queued")
    serve(monkeypatch, [row("a", 50.0, 1.0), queued])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    line = next(l for l in result.output.splitlines() if "queued one" in l)
    assert "50.00 ± 1.00" in line and gap_cell(result.output, "queued one") == "—"
    assert "heuristic" not in result.output


def test_default_output_is_json(monkeypatch):
    serve(monkeypatch, DAY_ONE)
    result = runner.invoke(main.app, ARGS)
    assert json.loads(result.output) == {"config_id": "runtime-v5", "rows": DAY_ONE}


def test_no_definitive_claims_for_chain(monkeypatch):
    serve(monkeypatch, [row("a", 0.0, 1.0), row("b", 2.0, 1.0, rank=2), row("c", 4.0, 1.0, rank=3)])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert gap_cell(out, "c  ") == "—"
    assert gap_cell(out, "b  ") == "-2.00 *"
    assert gap_cell(out, "a  ") == "-4.00"
    assert 4.0 > main.Z * (2 ** 0.5)
    assert not re.search(r"tie|separable|noise", out, re.I)
    assert len(out.splitlines()) == 4 + 1 + 2


def test_top_row_is_highest_score_not_first(monkeypatch):
    serve(monkeypatch, [row("a", 0.0, 10.0), row("b", 1.0, 0.1, rank=2), row("c", 5.0, 0.1, rank=3)])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert gap_cell(out, "c  ") == "—"
    assert gap_cell(out, "a  ") == "-5.00 *"
    assert gap_cell(out, "b  ") == "-4.00"


def test_zero_se_band(monkeypatch):
    serve(monkeypatch, [row("a", 100.0, 0.0), row("b", 99.0, 0.0, rank=2), row("c", 100.0, 0.0, rank=3)])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert gap_cell(out, "c  ") == "+0.00 *" and gap_cell(out, "b  ") == "-1.00"


def test_top_ignores_unscored_rows(monkeypatch):
    serve(monkeypatch, [row("big", 99.0, 1.0, state="queued"), row("badse", 98.0, float("nan"), rank=2),
                        row("a", 50.0, 1.0, rank=3), row("b", 49.0, 1.0, rank=4)])
    out = runner.invoke(main.app, ARGS + ["--table"]).output
    assert gap_cell(out, "big") == "—" and gap_cell(out, "badse") == "—"
    assert gap_cell(out, "a  ") == "—" and gap_cell(out, "b  ") == "-1.00 *"


def test_null_username_and_parent_rows(monkeypatch):
    serve(monkeypatch, [row("orig", 50.0, 1.0, parent_id=None),
                        row("anon", 40.0, 1.0, rank=2, submitter={"username": None}, parent_id="x")])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    lines = {l.split()[1]: l for l in result.output.splitlines()[1:3]}
    assert "original" in lines["orig"] and "—" in lines["anon"] and "original" not in lines["anon"]


def test_empty_board(monkeypatch):
    serve(monkeypatch, [])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0].split() == ["rank", "name", "submitter", "validation", "test", "±", "se", "gap", "to", "top"]
    assert "No standard errors reported." in result.output


def test_bad_numbers_are_unscored(monkeypatch):
    serve(monkeypatch, [row("nan", 50.0, float("nan")), row("neg", 50.0, -1.0, rank=2),
                        row("inf", float("inf"), 1.0, rank=3), row("ok", 50.0, 1.0, rank=4)])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    out = result.output
    lines = {l.split()[1]: l for l in out.splitlines()[1:5]}
    assert all(v not in l.lower().replace(k, "") for k, l in lines.items() for v in ("nan", "inf", "-1"))
    assert "separable" not in out and gap_cell(out, "ok  ") == "—"
    assert cells(lines["nan"])[-2] == "50.00" and cells(lines["neg"])[-2] == "50.00"
    assert cells(lines["inf"])[-2] == "— ± 1.00"


def test_name_and_rank_fallbacks(monkeypatch):
    serve(monkeypatch, [row(None, 50.0, 1.0, rank=0), row("", 40.0, 1.0, rank=None)])
    result = runner.invoke(main.app, ARGS + ["--table"])
    assert result.exit_code == 0
    first, second = result.output.splitlines()[1:3]
    assert first.split()[:2] == ["0", "—"] and second.split()[:2] == ["—", "—"]


def test_table_needs_model():
    result = runner.invoke(main.app, ["leaderboard", "--table"])
    assert result.exit_code == 2
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", result.output)
    text = re.sub(r"[\u2500-\u257f]", "", text)
    flat = re.sub(r"\s+", "", text)
    assert re.sub(r"\s+", "", "--table needs --model: rows are compared within one competition.") in flat


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
