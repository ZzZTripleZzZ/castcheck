"""Tests for the CLI wiring (DESIGN §5): run planning, skip rules, exit codes, run journal.

The commands themselves are thin, so what is worth testing is the *decisions* they make — which
runs to fetch, which to skip, and what exit code the workflow sees — none of which needs network or
parquet files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from typer.testing import CliRunner

from castcheck import __version__, cli
from castcheck.config import ModelSpec

runner = CliRunner()

IFS = ModelSpec(model_id="ifs_hres", family="ECMWF IFS HRES", source="ecmwf", product="oper",
                init_field=None, inits=(0, 12), step_h=6, max_h=240, native_extremes=())
GFS = ModelSpec(model_id="gfs", family="NCEP GFS", source="gfs", product="pgrb2.0p25",
                init_field=None, inits=(0, 12), step_h=6, max_h=240, native_extremes=())
GRAPH_GFS = ModelSpec(model_id="graphcast_gfs", family="GraphCast", source="aiwp", product="GRAP",
                      init_field="GFS", inits=(0, 12), step_h=6, max_h=240, native_extremes=())
GRAPH_IFS = ModelSpec(model_id="graphcast_ifs", family="GraphCast", source="aiwp", product="GRAP",
                      init_field="IFS", inits=(0, 12), step_h=6, max_h=240, native_extremes=())


@pytest.fixture(autouse=True)
def _journal_to_tmp(tmp_path, monkeypatch):
    """Keep every test's run journal out of the repo's data/ directory."""
    from castcheck import store

    monkeypatch.setattr(store, "LAST_RUN_PATH", tmp_path / "last_run.json")
    return tmp_path / "last_run.json"


def _no_history(*_args, **_kwargs):
    return set()


def _no_attempts(*_args, **_kwargs):
    return {}


# --------------------------------------------------------------------------- version


def test_version_prints_package_version():
    res = runner.invoke(cli.app, ["version"])
    assert res.exit_code == 0
    assert res.stdout.strip() == __version__


def test_version_matches_pyproject():
    import tomllib

    from castcheck.config import REPO_ROOT

    doc = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert doc["project"]["version"] == __version__


# --------------------------------------------------------------------------- availability delays


def test_availability_delay_is_per_source_and_per_aiwp_init_field():
    assert cli.availability_delay_h(GFS) == 5.5
    assert cli.availability_delay_h(IFS) == 8.0
    # AIWP publishes its GFS-initialised runs hours before the IFS-initialised ones
    assert cli.availability_delay_h(GRAPH_GFS) < cli.availability_delay_h(GRAPH_IFS)


# --------------------------------------------------------------------------- plan_runs


def test_plan_skips_runs_that_cannot_be_published_yet():
    now = datetime(2026, 8, 30, 6, 0, tzinfo=UTC)  # 06:00Z: GFS 00Z is out, ECMWF 00Z is not
    jobs = cli.plan_runs([GFS, IFS], now, lookback_days=0, have=_no_history, last_attempt=_no_attempts)
    assert [(m.model_id, i.hour) for m, i in jobs] == [("gfs", 0)]


def test_plan_covers_both_cycles_and_the_lookback_window():
    now = datetime(2026, 8, 30, 23, 0, tzinfo=UTC)
    jobs = cli.plan_runs([GFS], now, lookback_days=2, have=_no_history, last_attempt=_no_attempts)
    # 3 days x 2 cycles, all of them long past the 5.5 h delay
    assert len(jobs) == 6
    assert jobs == sorted(jobs, key=lambda j: j[1])
    assert {i.hour for _, i in jobs} == {0, 12}


def test_plan_treats_12z_exactly_like_00z():
    """A 12Z run becomes fetchable the same number of hours after its own initialisation."""
    now = datetime(2026, 8, 30, 20, 30, tzinfo=UTC)  # 12Z + 8.5 h
    jobs = cli.plan_runs([IFS], now, lookback_days=0, have=_no_history, last_attempt=_no_attempts)
    assert [i.hour for _, i in jobs] == [0, 12]
    earlier = cli.plan_runs([IFS], now.replace(hour=19), lookback_days=0,
                            have=_no_history, last_attempt=_no_attempts)
    assert [i.hour for _, i in earlier] == [0]


def test_plan_skips_runs_already_stored_complete():
    now = datetime(2026, 8, 30, 23, 0, tzinfo=UTC)
    stored = {pd.Timestamp("2026-08-30T00:00Z"), pd.Timestamp("2026-08-30T12:00Z")}
    jobs = cli.plan_runs([GFS], now, lookback_days=0, have=lambda *_: stored, last_attempt=_no_attempts)
    assert jobs == []


def test_plan_does_not_retry_a_partial_run_more_often_than_min_retry_h():
    """A run upstream never completes must not be re-downloaded by every scheduled pass."""
    now = datetime(2026, 8, 30, 23, 0, tzinfo=UTC)
    just_tried = {pd.Timestamp("2026-08-30T00:00Z"): pd.Timestamp("2026-08-30T22:00Z"),
                  pd.Timestamp("2026-08-30T12:00Z"): pd.Timestamp("2026-08-30T22:00Z")}
    jobs = cli.plan_runs([GFS], now, lookback_days=0, have=_no_history,
                         last_attempt=lambda *_: just_tried, min_retry_h=3.0)
    assert jobs == []
    # …but it is retried once the interval has passed, and only until it leaves the lookback window
    later = cli.plan_runs([GFS], now, lookback_days=0, have=_no_history,
                          last_attempt=lambda *_: just_tried, min_retry_h=0.5)
    assert len(later) == 2


def test_plan_passes_a_bounded_date_window_to_the_store():
    """The completeness lookup must be scoped, or it reads the whole archive on every invocation."""
    seen: list[tuple[str, str]] = []

    def record(_model, start, end):
        seen.append((start, end))
        return set()

    now = datetime(2026, 8, 30, 23, 0, tzinfo=UTC)
    cli.plan_runs([GFS], now, lookback_days=3, have=record, last_attempt=_no_attempts)
    assert seen == [("2026-08-27", "2026-08-30")]


# --------------------------------------------------------------------------- fetch-latest


def test_fetch_latest_reports_nothing_to_fetch(monkeypatch):
    monkeypatch.setattr(cli, "load_models", lambda: [GFS])
    monkeypatch.setattr(cli, "plan_runs", lambda *a, **k: [])
    res = runner.invoke(cli.app, ["fetch-latest"])
    assert res.exit_code == 0
    assert "nothing to fetch" in res.stdout


def test_fetch_latest_exits_nonzero_only_when_every_run_failed(monkeypatch):
    init = datetime(2026, 8, 30, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "load_models", lambda: [GFS])
    monkeypatch.setattr(cli, "plan_runs", lambda *a, **k: [(GFS, init), (GFS, init.replace(hour=12))])

    monkeypatch.setattr(cli, "_fetch_one", lambda m, i, s: (m.model_id, 0, 10))
    assert runner.invoke(cli.app, ["fetch-latest"]).exit_code == 1

    monkeypatch.setattr(cli, "_fetch_one", lambda m, i, s: (m.model_id, 0 if i.hour else 5520, 0))
    res = runner.invoke(cli.app, ["fetch-latest"])
    assert res.exit_code == 0  # one late upstream file is not a pipeline failure
    assert "present=5520" in res.stdout


def test_fetch_latest_survives_an_exception_in_one_run(monkeypatch):
    init = datetime(2026, 8, 30, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "load_models", lambda: [GFS])
    monkeypatch.setattr(cli, "plan_runs", lambda *a, **k: [(GFS, init), (GFS, init.replace(hour=12))])

    def flaky(m, i, _s):
        if i.hour == 0:
            raise RuntimeError("boom")
        return m.model_id, 5520, 0

    monkeypatch.setattr(cli, "_fetch_one", flaky)
    res = runner.invoke(cli.app, ["fetch-latest"])
    assert res.exit_code == 0
    assert "present=5520" in res.stdout


# --------------------------------------------------------------------------- backfill


def test_backfill_skips_stored_inits_and_scopes_the_lookup(monkeypatch):
    calls: dict = {}
    fetched: list[datetime] = []

    def fake_existing(model_id, start=None, end=None):
        calls.update(model_id=model_id, start=start, end=end)
        return {pd.Timestamp("2026-08-01T00:00Z")}

    monkeypatch.setattr("castcheck.store.existing_inits", fake_existing)
    monkeypatch.setattr(cli, "model_by_id", lambda _mid: GFS)
    monkeypatch.setattr(cli, "_fetch_one", lambda m, i, s: (fetched.append(i), (m.model_id, 1, 0))[1])

    res = runner.invoke(cli.app, ["backfill", "gfs", "2026-08-01", "2026-08-02"])
    assert res.exit_code == 0
    assert calls == {"model_id": "gfs", "start": "2026-08-01", "end": "2026-08-02"}
    assert sorted(i.isoformat() for i in fetched) == [
        "2026-08-01T12:00:00+00:00", "2026-08-02T00:00:00+00:00", "2026-08-02T12:00:00+00:00",
    ]


def test_backfill_with_skip_existing_disabled_refetches_everything(monkeypatch):
    fetched: list[datetime] = []
    monkeypatch.setattr("castcheck.store.existing_inits",
                        lambda *a, **k: pytest.fail("must not be consulted"))
    monkeypatch.setattr(cli, "model_by_id", lambda _mid: GFS)
    monkeypatch.setattr(cli, "_fetch_one", lambda m, i, s: (fetched.append(i), (m.model_id, 1, 0))[1])

    res = runner.invoke(cli.app, ["backfill", "gfs", "2026-08-01", "2026-08-01", "--no-skip-existing"])
    assert res.exit_code == 0
    assert len(fetched) == 2


# --------------------------------------------------------------------------- status


def test_status_exit_codes(monkeypatch):
    monkeypatch.setattr("castcheck.status.write_status", lambda _now: {"gaps_today": []})
    assert runner.invoke(cli.app, ["status"]).exit_code == 0

    monkeypatch.setattr("castcheck.status.write_status",
                        lambda _now: {"gaps_today": ["gfs 2026-08-30T00Z"]})
    res = runner.invoke(cli.app, ["status"])
    assert res.exit_code == 1
    assert "gaps today: 1" in res.stdout
    assert runner.invoke(cli.app, ["status", "--no-fail-on-gaps"]).exit_code == 0


# --------------------------------------------------------------------------- run journal


def test_command_records_its_outcome(_journal_to_tmp, monkeypatch):
    monkeypatch.setattr("castcheck.status.write_status", lambda _now: {"gaps_today": []})
    runner.invoke(cli.app, ["status"])

    doc = json.loads(_journal_to_tmp.read_text())
    entry = doc["commands"]["status"]
    assert entry["status"] == "ok"
    assert entry["exit_code"] == 0
    assert entry["last_success_at"] == entry["finished_at"]
    assert entry["castcheck_version"] == __version__
    assert "0 gap(s) today" in entry["summary"]


def test_journal_keeps_the_last_success_when_a_later_run_fails(_journal_to_tmp, monkeypatch):
    monkeypatch.setattr("castcheck.status.write_status", lambda _now: {"gaps_today": []})
    runner.invoke(cli.app, ["status"])
    good = json.loads(_journal_to_tmp.read_text())["commands"]["status"]["last_success_at"]

    monkeypatch.setattr("castcheck.status.write_status", lambda _now: {"gaps_today": ["x"]})
    runner.invoke(cli.app, ["status"])

    entry = json.loads(_journal_to_tmp.read_text())["commands"]["status"]
    assert entry["status"] == "error"
    assert entry["exit_code"] == 1
    assert entry["last_success_at"] == good


def test_json_log_format(monkeypatch, capsys):
    monkeypatch.setenv("CASTCHECK_LOG_JSON", "1")
    cli.setup_logging()
    cli.log.info("hello %s", "world")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert json.loads(line)["msg"] == "hello world"


# --------------------------------------------------------------------------- journal redaction


def test_redaction_removes_query_strings_tokens_and_absolute_paths():
    """`data/raw/last_run.json` is committed publicly, and exception text is not curated."""
    from castcheck.cli import redact
    from castcheck.config import REPO_ROOT

    out = redact("failed GET https://example.com/o.grib2?X-Amz-Signature=abc123&k=v")
    assert "X-Amz-Signature" not in out and "abc123" not in out
    assert "https://example.com/o.grib2" in out

    assert "sk-" not in redact("api_key=sk-0123456789abcdefghijklmno")
    assert "hunter2" not in redact("token: hunter2")

    assert redact(f"site written to {REPO_ROOT}/public") == "site written to public"
    assert "/Users/" not in redact("no such file: /Users/someone/secret/place/x.parquet")

    assert redact("  present=40   missing=0\n") == "present=40 missing=0"


def test_a_failing_command_journals_the_exception_type_not_its_raw_text(_journal_to_tmp, monkeypatch):
    from castcheck import cli, store

    def boom():
        raise RuntimeError("GET https://host/x?token=SUPERSECRETVALUE0123456789 failed")

    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    with pytest.raises(RuntimeError), cli._journal("fetch"):
        boom()

    entry = store.read_last_run()["commands"]["fetch"]
    assert entry["status"] == "error"
    assert entry["summary"].startswith("RuntimeError")
    assert "SUPERSECRETVALUE" not in entry["summary"]
