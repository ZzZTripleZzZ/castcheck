"""Tests for the daily data-gap alarm (`scripts/health_gaps.py`, DESIGN §7).

The decision this script makes is *when to wake a human up*, so the tests are about the boundaries:
a slot that is late but not late enough, a slot that is late because nobody has published it yet, a
months-old hole left by a backfill that is still running, and the recovery that has to close the
issue again. All of it runs against a synthetic status.json and a frozen clock — no data/, no
network, no repo state.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    """`scripts/` is not a package, so the module is loaded by path."""
    spec = importlib.util.spec_from_file_location("health_gaps", REPO_ROOT / "scripts" / "health_gaps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


health = _load()
NOW = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)


def _model_day(day: str, due_at: str, *, complete: bool = False, expected: bool = True) -> dict:
    return {"date": day, "due_at": due_at, "complete": complete, "expected": expected,
            "reason": "" if expected else "not_due_yet", "stations_complete": 0,
            "expected_steps": 40}


def _report(models=(), truth=(), truth_instant=()) -> dict:
    return {"as_of": "2026-09-01", "n_stations": 23, "schema_version": "0.3",
            "castcheck_version": "0.1.0", "models": list(models), "truth": list(truth),
            "truth_instant": list(truth_instant)}


def _model(model_id: str, init_hour: int, days: list[dict]) -> dict:
    return {"model_id": model_id, "init_hour": init_hour, "days": days}


# --------------------------------------------------------------------------- the 24 h boundary


def test_a_slot_overdue_by_less_than_a_day_is_not_an_alarm():
    """A run that missed this morning's pass is picked up by the next one — that is not news."""
    rep = _report([_model("gfs", 0, [_model_day("2026-09-01", "2026-09-01T05:30:00+00:00")])])
    assert health.overdue_gaps(rep, NOW) == []


def test_a_slot_overdue_by_more_than_a_day_is_an_alarm():
    rep = _report([_model("gfs", 0, [_model_day("2026-08-30", "2026-08-30T05:30:00+00:00")])])
    (gap,) = health.overdue_gaps(rep, NOW)
    assert gap["type"] == "model_run"
    assert gap["what"] == "gfs 00Z"
    assert gap["age_h"] == pytest.approx(55.0, abs=0.1)


def test_the_boundary_is_strictly_greater_than_max_age():
    """Exactly `max_age_h` old is still inside the grace period; a minute more is not."""
    exact = _report([_model("gfs", 0, [_model_day("2026-08-31", "2026-08-31T12:30:00+00:00")])])
    assert health.overdue_gaps(exact, NOW, max_age_h=24.0) == []

    older = _report([_model("gfs", 0, [_model_day("2026-08-31", "2026-08-31T12:29:00+00:00")])])
    assert len(health.overdue_gaps(older, NOW, max_age_h=24.0)) == 1
    assert health.overdue_gaps(older, NOW, max_age_h=48.0) == []


def test_a_complete_or_unexpected_slot_is_never_a_gap():
    """`expected: false` is the schedule saying nobody has published this yet (DESIGN §7)."""
    days = [
        _model_day("2026-08-01", "2026-08-01T05:30:00+00:00", complete=True),
        _model_day("2026-08-02", "2026-08-02T05:30:00+00:00", expected=False),
    ]
    assert health.overdue_gaps(_report([_model("gfs", 0, days)]), NOW) == []


# --------------------------------------------------------------------------- the other two tables


def test_a_missing_final_cli_report_is_a_truth_gap():
    rep = _report(truth=[{"station_id": "KOKC", "days": [
        {"date": "2026-08-28", "due_at": "2026-08-29T10:00:00+00:00",
         "cli_final": False, "expected": True}]}])
    (gap,) = health.overdue_gaps(rep, NOW)
    assert gap["type"] == "truth" and gap["what"] == "KOKC"


def test_an_incomplete_instant_day_uses_the_schedule_modules_deadline():
    """`truth_instant` rows are the one grid without a `due_at`, so it comes from castcheck.schedule."""
    from castcheck import schedule

    rep = _report(truth_instant=[{"station_id": "KBOS", "days": [
        {"date": "2026-08-28", "n_instants": 3, "complete": False, "expected": True}]}])
    (gap,) = health.overdue_gaps(rep, NOW)
    assert gap["type"] == "truth_instant"
    assert gap["due_at"] == schedule.instant_due_at(date(2026, 8, 28)).isoformat()
    assert "3/4" in gap["detail"]


# --------------------------------------------------------------------------- backfill vs outage


def test_old_holes_are_reported_but_do_not_raise_the_alarm():
    """An in-flight backfill leaves months of empty slots; alerting on them means alerting always."""
    rep = _report([_model("aurora_ifs", 0, [
        _model_day("2026-08-31", "2026-08-31T09:30:00+00:00"),      # 27 h — the outage
        _model_day("2026-06-01", "2026-06-01T09:30:00+00:00"),      # 3 months — the backfill
    ])])
    alerting, archive = health.split_recent(health.overdue_gaps(rep, NOW), lookback_days=7)
    assert [g["date"] for g in alerting] == ["2026-08-31"]
    assert [g["date"] for g in archive] == ["2026-06-01"]


def test_the_body_names_the_archive_count_even_when_nothing_is_alerting():
    rep = _report([_model("aurora_ifs", 0, [_model_day("2026-06-01", "2026-06-01T09:30:00+00:00")])])
    alerting, archive = health.split_recent(health.overdue_gaps(rep, NOW))
    body = health.markdown(alerting, archive, rep, NOW, 24.0, 7.0)
    assert "keeping up" in body
    assert "1 older slot(s)" in body


def test_the_body_is_worst_first_and_names_what_is_missing():
    rep = _report([
        _model("gfs", 0, [_model_day("2026-08-31", "2026-08-31T05:30:00+00:00")]),
        _model("ifs_hres", 12, [_model_day("2026-08-28", "2026-08-28T20:00:00+00:00")]),
    ])
    rows, archive = health.split_recent(health.overdue_gaps(rep, NOW))
    body = health.markdown(rows, archive, rep, NOW, 24.0, 7.0)
    assert body.index("ifs_hres 12Z") < body.index("gfs 00Z")   # 88 h before 31 h
    assert "2 slot(s) overdue" in body


# --------------------------------------------------------------------------- the CLI contract


def _write(tmp_path: Path, report: dict) -> Path:
    p = tmp_path / "status.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


def test_exit_code_is_2_when_something_is_overdue_and_0_when_recovered(tmp_path, capsys):
    """`health.yml` branches on this: 0 closes the issue, 2 opens or updates it."""
    stuck = _report([_model("gfs", 0, [_model_day("2026-08-30", "2026-08-30T05:30:00+00:00")])])
    args = ["--status-json", str(_write(tmp_path, stuck)), "--now", NOW.isoformat()]
    assert health.main(args) == 2

    recovered = _report([_model("gfs", 0, [
        _model_day("2026-08-30", "2026-08-30T05:30:00+00:00", complete=True)])])
    assert health.main(["--status-json", str(_write(tmp_path, recovered)),
                        "--now", NOW.isoformat()]) == 0
    assert "keeping up" in capsys.readouterr().out


def test_a_missing_status_json_is_an_error_not_an_all_clear(tmp_path):
    """Exit 1, so the workflow fails loudly instead of quietly closing the outage issue."""
    assert health.main(["--status-json", str(tmp_path / "nope.json")]) == 1


def test_github_output_carries_the_count(tmp_path, monkeypatch):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    rep = _report([_model("gfs", 0, [
        _model_day("2026-08-30", "2026-08-30T05:30:00+00:00"),
        _model_day("2026-08-30", "2026-08-30T17:30:00+00:00"),
    ])])
    health.main(["--status-json", str(_write(tmp_path, rep)), "--now", NOW.isoformat(),
                 "--github-output"])
    assert out.read_text().strip() == "overdue=2"
