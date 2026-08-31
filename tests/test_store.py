"""Tests for `castcheck.store`: shard layout, upsert rules, and the scoped completeness reads.

The scalability contract matters as much as the values: after a year of backfill one model holds
millions of rows, so `existing_inits` must open only the shards in the requested window and read
only the columns it needs. Both are asserted by recording what actually reaches `pyarrow`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

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


# --------------------------------------------------------------------------- staging files


def _fv_dir(data_dir) -> object:
    return data_dir / "forecast_values" / f"model_id={MODEL}"


def test_a_half_written_staging_file_does_not_break_the_readers(data_dir):
    """A crashed or concurrent writer leaves a staging file behind; readers must not open it.

    The old staging name was ``<shard>.tmp.parquet``, which every ``*.parquet`` glob matched — so
    an interrupted write in one command made an unrelated command fail with ``ArrowInvalid`` on a
    truncated file. The name is now a dotfile that does not end in ``.parquet``, and the shard
    listing skips both shapes regardless.
    """
    store.upsert_forecast_values(_values(_init(1)))
    store.upsert_truth_instant(_instant_rows())
    for junk in (_fv_dir(data_dir) / ".year_month=2026-08.parquet.99.tmp",
                 _fv_dir(data_dir) / "year_month=2026-08.tmp.parquet",
                 _fv_dir(data_dir) / "._year_month=2026-08.parquet",
                 data_dir / "truth_instant" / ".year=2026.parquet.99.tmp"):
        junk.write_bytes(b"not a parquet file at all")
    # a file-sync conflict copy: valid parquet, and a duplicate of every row in the real shard
    conflict = _fv_dir(data_dir) / "year_month=2026-08 2.parquet"
    conflict.write_bytes((_fv_dir(data_dir) / "year_month=2026-08.parquet").read_bytes())

    assert len(store.read_forecast_values()) == STEPS
    assert store.existing_inits(MODEL) == {pd.Timestamp(_init(1))}
    assert len(store.read_truth_instant()) == 1
    assert [p.name for p in store.forecast_value_shards()] == ["year_month=2026-08.parquet"]


def test_the_staging_file_is_removed_by_a_successful_write(data_dir):
    store.upsert_forecast_values(_values(_init(1)))
    assert [p.name for p in _fv_dir(data_dir).iterdir()] == ["year_month=2026-08.parquet"]


# --------------------------------------------------------------------------- schema growth


def test_a_shard_written_before_a_column_existed_still_reads(data_dir):
    """`native_overhang_h` and the other v0.3 columns arrived after the first year of data."""
    store.write_daily(_daily(_init(1), "2026-08-02"))
    path = store.daily_path(MODEL, 2026)
    old = pq.read_table(path).to_pandas().drop(columns=["native_overhang_h", "n_obs_samples"])
    store._write(old, path)

    df = store.read_daily()
    assert list(df.columns) == store.DAILY_COLUMNS
    assert df["native_overhang_h"].isna().all()


# --------------------------------------------------------------------------- f000


def _aiwp_values(init: datetime) -> pd.DataFrame:
    """An AIWP-shaped run: f000 plus 39 forecast steps, i.e. one step short of complete."""
    df = _values(init, n_steps=STEPS - 1, model_id="aiwp")
    zero = df.iloc[[0]].copy()
    zero["lead_h"] = 0
    zero["valid_time"] = zero["init_time"]
    return pd.concat([zero, df], ignore_index=True)


def test_the_analysis_step_does_not_count_towards_completeness(data_dir, monkeypatch):
    """AIWP stores f000 and the GRIB sources do not; counting it declared a short run complete."""
    store.upsert_forecast_values(_aiwp_values(_init(1)))
    assert store.existing_inits("aiwp", min_valid=STEPS) == set()

    store.upsert_forecast_values(_values(_init(1), model_id="aiwp"))   # the missing step arrives
    assert store.existing_inits("aiwp", min_valid=STEPS) == {pd.Timestamp(_init(1))}


# --------------------------------------------------------------------------- daily_forecasts


def _daily(init: datetime, climo_date: str, model_id: str = MODEL, tmax: float = 30.0) -> pd.DataFrame:
    row = {
        "model_id": model_id, "model_version": "v1", "init_time": init, "station_id": "KNYC",
        "climo_date": date.fromisoformat(climo_date), "lead_day": 1, "method": "bilinear",
        "tmax_sampled_c": tmax, "tmin_sampled_c": 20.0, "n_samples": 4,
        "tmax_native_c": float("nan"), "tmin_native_c": float("nan"), "missing_reason": "",
        "schema_version": "0.3", "methodology_version": "0.3",
    }
    return pd.DataFrame([row])


def test_writing_one_day_does_not_delete_the_rest_of_the_year(data_dir):
    """`derive` takes a date window in v0.3; an overwrite would drop every other day of the year."""
    store.write_daily(_daily(_init(1), "2026-08-02"))
    store.write_daily(_daily(_init(2), "2026-08-03"))

    df = store.read_daily()
    assert sorted(str(d) for d in df["climo_date"]) == ["2026-08-02", "2026-08-03"]


def test_re_deriving_a_day_replaces_it_rather_than_duplicating_it(data_dir):
    store.write_daily(_daily(_init(1), "2026-08-02", tmax=30.0))
    store.write_daily(_daily(_init(1), "2026-08-02", tmax=31.5))

    df = store.read_daily()
    assert len(df) == 1
    assert df["tmax_sampled_c"].iloc[0] == pytest.approx(31.5)


# --------------------------------------------------------------------------- truth_instant


def _instant_rows(station_id: str = "KNYC", when: str = "2026-08-01T12", temp: float = 20.0,
                  source: str = "ASOS_IEM") -> pd.DataFrame:
    t = pd.Timestamp(when, tz="UTC")
    df = pd.DataFrame([{
        "station_id": station_id, "valid_time": t, "temp_c": temp,
        "obs_time": t - pd.Timedelta(minutes=9), "source": source, "n_reports": 1, "qc_flag": "",
        "schema_version": "0.3", "methodology_version": "0.3",
    }], columns=store.TRUTH_INSTANT_COLUMNS)
    df["temp_c"] = df["temp_c"].astype("float32")
    df["n_reports"] = df["n_reports"].astype("int8")
    return df


def test_truth_instant_is_sharded_by_the_year_of_the_instant(data_dir):
    out = store.upsert_truth_instant(pd.concat([_instant_rows(when="2025-12-31T18"),
                                                _instant_rows(when="2026-01-01T00")]))
    assert sorted(out) == ["truth_instant/year=2025.parquet", "truth_instant/year=2026.parquet"]
    assert store.truth_instant_path(2026) == data_dir / "truth_instant" / "year=2026.parquet"


def test_truth_instant_dtypes_stay_narrow_on_disk(data_dir):
    store.upsert_truth_instant(_instant_rows())
    df = pq.read_table(store.truth_instant_path(2026)).to_pandas()
    assert str(df["temp_c"].dtype) == "float32"
    assert str(df["n_reports"].dtype) == "int8"


def test_reading_one_year_of_truth_instant(data_dir):
    store.upsert_truth_instant(pd.concat([_instant_rows(when="2025-12-31T18"),
                                          _instant_rows(when="2026-01-01T00")]))
    assert len(store.read_truth_instant(years=[2026])) == 1
    assert len(store.read_truth_instant()) == 2
