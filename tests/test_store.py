"""Tests for `castcheck.store`: shard layout, upsert rules, and the scoped completeness reads.

The scalability contract matters as much as the values: after a year of backfill one model holds
millions of rows, so `existing_inits` must open only the shards in the requested window and read
only the columns it needs. Both are asserted by recording what actually reaches `pyarrow`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest

from castcheck import store
from castcheck.sources.base import FORECAST_VALUE_COLUMNS

MODEL = "gfs"
STEPS = 40  # max_h / step_h for every model in models.yaml


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def _values(init: datetime, n_steps: int = STEPS, missing_after: int | None = None,
            fetched_at: datetime | None = None, model_id: str = MODEL) -> pd.DataFrame:
    """One station's `t2` rows for a run; steps beyond `missing_after` are explicit missing rows."""
    rows = []
    for i in range(1, n_steps + 1):
        step = 6 * i
        absent = missing_after is not None and i > missing_after
        rows.append({
            "model_id": model_id, "model_version": "gfs-0p25", "init_time": init,
            "valid_time": init + timedelta(hours=step), "lead_h": step, "station_id": "KNYC",
            "variable": "t2", "bucket_h": 0, "method": "bilinear",
            "value_c": float("nan") if absent else 20.0 + i,
            "missing_reason": "http_503" if absent else "",
            "source_url": "https://example.com/x", "fetched_at": fetched_at or init + timedelta(hours=6),
            "schema_version": "0.1", "methodology_version": "0.1",
        })
    return pd.DataFrame(rows, columns=FORECAST_VALUE_COLUMNS)


def _init(day: int, hour: int = 0, month: int = 8) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=UTC)


# --------------------------------------------------------------------------- shards and upsert


def test_rows_are_sharded_by_init_month(data_dir):
    out = store.upsert_forecast_values(pd.concat([_values(_init(30, month=7)), _values(_init(1, month=8))]))
    assert sorted(out) == ["forecast_values/model_id=gfs/year_month=2026-07.parquet",
                           "forecast_values/model_id=gfs/year_month=2026-08.parquet"]


def test_upsert_is_idempotent_and_a_value_replaces_a_missing_row(data_dir):
    init = _init(1)
    store.upsert_forecast_values(_values(init, missing_after=10))
    store.upsert_forecast_values(_values(init, missing_after=10))
    assert len(store.read_forecast_values()) == STEPS

    later = _values(init, fetched_at=init + timedelta(hours=12))
    store.upsert_forecast_values(later)
    df = store.read_forecast_values()
    assert len(df) == STEPS
    assert (df["missing_reason"] == "").all()  # present beats missing


def test_a_missing_row_never_overwrites_a_stored_value(data_dir):
    init = _init(1)
    store.upsert_forecast_values(_values(init))
    store.upsert_forecast_values(_values(init, missing_after=5, fetched_at=init + timedelta(hours=24)))
    df = store.read_forecast_values()
    assert (df["missing_reason"] == "").all()


# --------------------------------------------------------------------------- existing_inits


def test_existing_inits_accepts_only_complete_runs(data_dir):
    store.upsert_forecast_values(_values(_init(1)))                      # complete
    store.upsert_forecast_values(_values(_init(2), missing_after=39))    # one step short
    assert store.existing_inits(MODEL) == {pd.Timestamp(_init(1))}


def test_existing_inits_ignores_other_models(data_dir):
    store.upsert_forecast_values(_values(_init(1), model_id="ifs_hres"))
    assert store.existing_inits(MODEL) == set()
    assert store.existing_inits("ifs_hres") == {pd.Timestamp(_init(1))}


def test_existing_inits_opens_only_the_shards_in_the_window(data_dir, monkeypatch):
    for month in (6, 7, 8):
        store.upsert_forecast_values(_values(_init(1, month=month)))

    opened: list[str] = []
    real = pq.read_table
    monkeypatch.setattr(pq, "read_table", lambda p, **kw: (opened.append(str(p)), real(p, **kw))[1])

    got = store.existing_inits(MODEL, start="2026-08-01", end="2026-08-31")

    assert got == {pd.Timestamp(_init(1, month=8))}
    assert len(opened) == 1 and "2026-08" in opened[0]


def test_existing_inits_reads_only_the_columns_it_needs(data_dir, monkeypatch):
    store.upsert_forecast_values(_values(_init(1)))
    seen: list = []
    real = pq.read_table
    monkeypatch.setattr(pq, "read_table",
                        lambda p, **kw: (seen.append(kw.get("columns")), real(p, **kw))[1])

    store.existing_inits(MODEL, start="2026-08-01", end="2026-08-31")

    assert seen and seen[0] is not None
    assert set(seen[0]) <= set(store._COMPLETENESS_COLUMNS)
    assert "value_c" not in seen[0] and "source_url" not in seen[0]


def test_read_forecast_values_projects_columns_but_keeps_init_time_for_the_range(data_dir):
    store.upsert_forecast_values(_values(_init(1)))
    df = store.read_forecast_values([MODEL], start="2026-08-01", end="2026-08-31", columns=["variable"])
    assert set(df.columns) == {"variable", "init_time"}
    plain = store.read_forecast_values([MODEL], columns=["variable"])
    assert list(plain.columns) == ["variable"]


def test_forecast_value_shards_filters_by_month(data_dir):
    for month in (6, 7, 8):
        store.upsert_forecast_values(_values(_init(1, month=month)))
    got = [p.name for p in store.forecast_value_shards([MODEL], start="2026-07-01", end="2026-07-31")]
    assert got == ["year_month=2026-07.parquet"]


def test_existing_inits_on_an_empty_store_is_empty(data_dir):
    assert store.existing_inits(MODEL) == set()
    assert store.last_attempt_by_init(MODEL) == {}


# --------------------------------------------------------------------------- last_attempt_by_init


def test_last_attempt_reports_the_newest_fetch_per_init(data_dir):
    init = _init(1)
    store.upsert_forecast_values(_values(init, missing_after=10, fetched_at=_init(1, 6)))
    store.upsert_forecast_values(_values(init, missing_after=10, fetched_at=_init(1, 9)))
    store.upsert_forecast_values(_values(_init(2), fetched_at=_init(2, 6)))

    attempts = store.last_attempt_by_init(MODEL, start="2026-08-01", end="2026-08-31")
    assert attempts[pd.Timestamp(init)] == pd.Timestamp(_init(1, 9))
    assert attempts[pd.Timestamp(_init(2))] == pd.Timestamp(_init(2, 6))


# --------------------------------------------------------------------------- dtypes


def test_cast_narrows_a_float_column_with_missing_values_to_a_nullable_int():
    """`float64` with NaN cannot go straight to Int16; the fallback must not leave the column wide."""
    df = pd.DataFrame({"tmax_f": [76.0, float("nan"), 64.0]})
    out = store._cast(df, {"tmax_f": "Int16"})
    assert str(out["tmax_f"].dtype) == "Int16"
    assert out["tmax_f"].tolist()[0] == 76
    assert pd.isna(out["tmax_f"].tolist()[1])


def test_cast_logs_and_keeps_the_column_when_it_really_cannot_narrow(caplog):
    import logging

    df = pd.DataFrame({"tmax_f": ["not a number"]})
    with caplog.at_level(logging.WARNING, logger="castcheck.store"):
        out = store._cast(df, {"tmax_f": "Int16"})
    assert out["tmax_f"].tolist() == ["not a number"]   # data preserved, never coerced to NA
    assert str(out["tmax_f"].dtype) != "Int16"
    assert any("tmax_f" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- run journal


def test_record_run_is_atomic_and_keeps_one_entry_per_command(data_dir, monkeypatch):
    monkeypatch.setattr(store, "LAST_RUN_PATH", data_dir / "raw" / "last_run.json")
    store.record_run("fetch-latest", status="ok", summary="3/3 run(s) with data", duration_s=12.0)
    store.record_run("truth", status="error", summary="boom", exit_code=1)

    doc = store.read_last_run()
    assert set(doc["commands"]) == {"fetch-latest", "truth"}
    assert doc["commands"]["fetch-latest"]["duration_s"] == 12.0
    assert doc["commands"]["truth"]["last_success_at"] is None
    assert list((data_dir / "raw").glob(".last_run.*")) == []  # temp file cleaned up


def test_read_last_run_treats_a_corrupt_file_as_empty(data_dir, monkeypatch):
    path = data_dir / "raw" / "last_run.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ not json")
    monkeypatch.setattr(store, "LAST_RUN_PATH", path)
    assert store.read_last_run() == {"commands": {}}
