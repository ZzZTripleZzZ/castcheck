"""Tests for the ``truth_instant`` path: IEM parsing, instant selection, QC flags, upsert, merge.

The fixture ``iem_asos_knyc_20260801.csv`` is a verbatim response from the IEM ASOS archive
(``station=NYC&data=tmpf&report_type=3``, 2026-08-01 UTC): 24 routine reports at :51.

Most of the file builds report series by hand instead, because what has to be pinned down is the
*selection rule* — which report represents 12 UTC when several are available, and what is recorded
when none is — and a hand-built series is the only way to put a report at an awkward minute.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from castcheck import merge, store
from castcheck.config import Station
from castcheck.sources import iem_asos
from castcheck.store import TRUTH_INSTANT_COLUMNS
from castcheck.truth_instant import (
    SOURCE_IEM,
    SOURCE_NWS,
    instant_from_reports,
    synoptic_times,
)

FIXTURES = Path(__file__).parent / "fixtures"

KNYC = Station(id="KNYC", name="New York Central Park", cli_pil="CLINYC", tz="America/New_York",
               std_offset_h=-5, lat=40.78333, lon=-73.96667, elev_m=46.9, market_city="NYC",
               iem_id="NYC", grid_elev_m=11.4)
KMSP = Station(id="KMSP", name="Minneapolis-St Paul", cli_pil="CLIMSP", tz="America/Chicago",
               std_offset_h=-6, lat=44.88306, lon=-93.22889, elev_m=256.0)


def reports(*pairs) -> pd.DataFrame:
    """``("2026-08-01T11:51", 20.0)`` pairs -> the frame ``instant_from_reports`` consumes."""
    return pd.DataFrame({
        "obs_time": pd.to_datetime([p[0] for p in pairs], utc=True),
        "temp_c": np.array([p[1] for p in pairs], dtype="float64"),
    })


def at(hour: int, day: str = "2026-08-01") -> pd.Timestamp:
    return pd.Timestamp(f"{day}T{hour:02d}:00", tz="UTC")


# --------------------------------------------------------------------------- IEM parsing


def test_parse_the_archive_csv():
    df = iem_asos.parse_asos_csv((FIXTURES / "iem_asos_knyc_20260801.csv").read_text())
    assert len(df) == 24
    assert list(df.columns) == ["obs_time", "temp_c", "report_type"]
    assert df["obs_time"].iloc[0] == pd.Timestamp("2026-08-01T00:51", tz="UTC")
    assert (df["obs_time"].dt.minute == 51).all()
    assert df["temp_c"].iloc[0] == pytest.approx((75.0 - 32.0) * 5 / 9, abs=1e-4)
    assert (df["report_type"] == 3).all()
    assert str(df["temp_c"].dtype) == "float32"


def test_missing_temperature_keeps_its_row_as_nan():
    df = iem_asos.parse_asos_csv("station,valid,tmpf\nNYC,2026-08-01 11:51,M\nNYC,2026-08-01 12:51,70.00\n")
    assert len(df) == 2
    assert np.isnan(df["temp_c"].iloc[0])
    assert df["temp_c"].iloc[1] == pytest.approx(21.111, abs=1e-3)


def test_an_unparsable_row_is_dropped_not_fatal():
    df = iem_asos.parse_asos_csv("station,valid,tmpf\nNYC,not-a-time,70.00\nNYC,2026-08-01 12:51,70.00\n")
    assert len(df) == 1


def test_the_throttle_notice_is_not_mistaken_for_an_empty_month():
    """IEM serves ``Too many requests`` with HTTP 200; treating it as data is a silent hole."""
    assert iem_asos.looks_like_csv("station,valid,tmpf\nNYC,2026-08-01 00:51,75.00\n")
    assert not iem_asos.looks_like_csv("Too many requests from your IP address, slow down.")
    assert not iem_asos.looks_like_csv("<html><body>error</body></html>")


def test_iem_id_drops_the_leading_k_unless_one_is_frozen():
    assert iem_asos.iem_id(KNYC) == "NYC"          # frozen in stations.yaml
    assert iem_asos.iem_id(KMSP) == "MSP"          # derived
    assert iem_asos.iem_id(Station(id="PANC", name="x", cli_pil="CLIANC", tz="UTC", std_offset_h=0,
                                   lat=None, lon=None, elev_m=None)) == "PANC"


def test_month_chunks_split_on_calendar_boundaries():
    chunks = iem_asos._month_chunks(datetime(2024, 1, 15, tzinfo=UTC), datetime(2024, 3, 2, tzinfo=UTC))
    assert chunks == [(date(2024, 1, 15), date(2024, 1, 31)),
                      (date(2024, 2, 1), date(2024, 2, 29)),
                      (date(2024, 3, 1), date(2024, 3, 2))]


def test_the_request_end_bound_is_the_day_after_the_last_wanted_day():
    url = iem_asos._url("NYC", date(2026, 8, 1), date(2026, 8, 31), 3)
    assert "year2=2026&month2=9&day2=1" in url   # IEM's end bound is exclusive
    assert "report_type=3" in url and "tz=Etc/UTC" in url


# --------------------------------------------------------------------------- instant selection


def test_the_report_closest_to_the_hour_wins():
    obs = reports(("2026-08-01T11:35", 10.0), ("2026-08-01T11:51", 12.0), ("2026-08-01T12:20", 14.0))
    row = instant_from_reports(obs, KNYC, [at(12)]).iloc[0]
    assert row.temp_c == pytest.approx(12.0)
    assert row.obs_time == pd.Timestamp("2026-08-01T11:51", tz="UTC")
    assert row.n_reports == 3
    assert row.qc_flag == ""
    assert row.source == SOURCE_IEM


def test_a_scheduled_report_beats_a_closer_unscheduled_one():
    """A SPECI at :58 is nearer 12 UTC than the routine METAR at :51, and must still lose."""
    obs = reports(("2026-08-01T11:51", 12.0), ("2026-08-01T11:58", 12.9))
    row = instant_from_reports(obs, KNYC, [at(12)]).iloc[0]
    assert row.obs_time == pd.Timestamp("2026-08-01T11:51", tz="UTC")


def test_a_report_just_outside_the_window_is_not_used():
    obs = reports(("2026-08-01T11:24", 12.0), ("2026-08-01T05:51", 8.0))
    row = instant_from_reports(obs, KNYC, [at(12)]).iloc[0]
    assert np.isnan(row.temp_c)
    assert pd.isna(row.obs_time)
    assert row.n_reports == 0
    assert row.qc_flag == "gap_gt35min"   # the station is reporting, just not near the hour


def test_a_report_just_inside_the_window_is_used():
    obs = reports(("2026-08-01T11:26", 12.0))
    row = instant_from_reports(obs, KNYC, [at(12)]).iloc[0]
    assert row.temp_c == pytest.approx(12.0)
    assert row.qc_flag == ""


def test_silence_is_no_report_not_a_gap():
    row = instant_from_reports(reports(), KNYC, [at(12)]).iloc[0]
    assert np.isnan(row.temp_c) and row.qc_flag == "no_report" and row.n_reports == 0


def test_a_report_without_a_temperature_is_counted_but_not_used():
    obs = reports(("2026-08-01T11:51", float("nan")))
    row = instant_from_reports(obs, KNYC, [at(12)]).iloc[0]
    assert np.isnan(row.temp_c)
    assert row.n_reports == 1          # the station did report
    assert row.qc_flag == "no_report"  # but nothing usable anywhere near


def test_an_eight_degree_jump_within_an_hour_is_suspect():
    obs = reports(("2026-08-01T10:51", 12.0), ("2026-08-01T11:51", 25.0), ("2026-08-01T12:51", 13.0))
    row = instant_from_reports(obs, KNYC, [at(12)]).iloc[0]
    assert row.temp_c == pytest.approx(25.0)
    assert row.qc_flag == "suspect"


def test_a_steep_but_plausible_change_is_not_suspect():
    obs = reports(("2026-08-01T10:51", 12.0), ("2026-08-01T11:51", 19.0), ("2026-08-01T12:51", 21.0))
    assert instant_from_reports(obs, KNYC, [at(12)]).iloc[0].qc_flag == ""


def test_a_value_with_no_neighbours_is_not_flagged_suspect():
    obs = reports(("2026-08-01T11:51", 19.0))
    assert instant_from_reports(obs, KNYC, [at(12)]).iloc[0].qc_flag == ""


def test_every_requested_instant_gets_a_row_in_the_declared_schema():
    obs = iem_asos.parse_asos_csv((FIXTURES / "iem_asos_knyc_20260801.csv").read_text())
    out = instant_from_reports(obs, KNYC, synoptic_times(date(2026, 8, 1), date(2026, 8, 1)))
    assert list(out.columns) == TRUTH_INSTANT_COLUMNS
    assert len(out) == 4
    assert list(out["valid_time"].dt.hour) == [0, 6, 12, 18]
    # the fixture is one UTC day, so 00 UTC has no report before it: its nearest is 00:51, 51 min
    # out, which is exactly the case ``gap_gt35min`` exists to name
    assert list(out["qc_flag"]) == ["gap_gt35min", "", "", ""]
    assert out["temp_c"].iloc[1:].notna().all()
    assert (out["station_id"] == "KNYC").all()
    assert str(out["temp_c"].dtype) == "float32" and str(out["n_reports"].dtype) == "int8"


def test_synoptic_times_covers_whole_utc_days():
    idx = synoptic_times(date(2026, 8, 1), date(2026, 8, 3))
    assert len(idx) == 12
    assert idx[0] == pd.Timestamp("2026-08-01T00", tz="UTC")
    assert idx[-1] == pd.Timestamp("2026-08-03T18", tz="UTC")


def test_selection_is_not_thrown_off_by_the_microsecond_timestamp_unit():
    """pandas parses to ``datetime64[us]``; Timestamp.value is ns. Mixing the two loses every row."""
    obs = reports(("2026-08-01T11:51", 12.0))
    obs["obs_time"] = obs["obs_time"].astype("datetime64[us, UTC]")
    assert instant_from_reports(obs, KNYC, [at(12)]).iloc[0].temp_c == pytest.approx(12.0)


# --------------------------------------------------------------------------- store and merge


def instant_rows(*specs) -> pd.DataFrame:
    """``(station, "2026-08-01T12", temp_or_None, source)`` tuples -> a truth_instant frame."""
    recs = []
    for station_id, when, temp, source in specs:
        t = pd.Timestamp(when, tz="UTC")
        recs.append({
            "station_id": station_id, "valid_time": t,
            "temp_c": float("nan") if temp is None else float(temp),
            "obs_time": pd.NaT if temp is None else t - pd.Timedelta(minutes=9),
            "source": source, "n_reports": 0 if temp is None else 1,
            "qc_flag": "no_report" if temp is None else "",
            "schema_version": "0.3", "methodology_version": "0.3",
        })
    df = pd.DataFrame.from_records(recs, columns=TRUTH_INSTANT_COLUMNS)
    df["temp_c"] = df["temp_c"].astype("float32")
    df["n_reports"] = df["n_reports"].astype("int8")
    return df


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def test_upsert_is_idempotent_and_shards_by_year(data_dir):
    rows = instant_rows(("KNYC", "2025-12-31T18", 1.0, SOURCE_IEM),
                        ("KNYC", "2026-01-01T00", 2.0, SOURCE_IEM))
    assert sorted(store.upsert_truth_instant(rows)) == ["truth_instant/year=2025.parquet",
                                                        "truth_instant/year=2026.parquet"]
    store.upsert_truth_instant(rows)
    assert len(store.read_truth_instant()) == 2


def test_the_archive_replaces_the_api_for_the_same_instant(data_dir):
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", 20.0, SOURCE_NWS)))
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", 20.6, SOURCE_IEM)))
    df = store.read_truth_instant()
    assert len(df) == 1
    assert df.iloc[0].source == SOURCE_IEM and df.iloc[0].temp_c == pytest.approx(20.6, abs=1e-3)


def test_the_api_never_replaces_the_archive(data_dir):
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", 20.6, SOURCE_IEM)))
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", 20.0, SOURCE_NWS)))
    assert store.read_truth_instant().iloc[0].source == SOURCE_IEM


def test_an_archive_gap_never_erases_an_api_value(data_dir):
    """The archive wins on authority, not on emptiness: a hole must not overwrite an observation."""
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", 20.0, SOURCE_NWS)))
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", None, SOURCE_IEM)))
    row = store.read_truth_instant().iloc[0]
    assert row.source == SOURCE_NWS and row.temp_c == pytest.approx(20.0)


def test_a_recovered_value_replaces_a_stored_gap(data_dir):
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", None, SOURCE_IEM)))
    store.upsert_truth_instant(instant_rows(("KNYC", "2026-08-01T12", 20.6, SOURCE_IEM)))
    assert store.read_truth_instant().iloc[0].temp_c == pytest.approx(20.6, abs=1e-3)


def test_read_returns_the_declared_schema_when_nothing_is_stored(data_dir):
    assert list(store.read_truth_instant().columns) == TRUTH_INSTANT_COLUMNS


def test_merge_unions_two_writers_with_the_same_precedence():
    ours = instant_rows(("KNYC", "2026-08-01T12", 20.0, SOURCE_NWS),
                        ("KNYC", "2026-08-01T18", None, SOURCE_NWS))
    theirs = instant_rows(("KNYC", "2026-08-01T12", 20.6, SOURCE_IEM),
                          ("KORD", "2026-08-01T12", 25.0, SOURCE_IEM))
    out = merge.merge_frames("truth_instant", ours, theirs)
    assert len(out) == 3                                    # neither side's rows are lost
    knyc12 = out[(out.station_id == "KNYC") & (out.valid_time.dt.hour == 12)].iloc[0]
    assert knyc12.source == SOURCE_IEM                      # archive wins the collision
    assert set(out.station_id) == {"KNYC", "KORD"}


def test_merge_recognises_the_new_shard_kind():
    assert merge.kind_of(Path("data/truth_instant/year=2026.parquet")) == "truth_instant"


# --------------------------------------------------------------------------- source wiring


def test_the_daily_command_asks_for_exactly_that_days_four_instants(monkeypatch):
    """`--date D` must mean D's 00/06/12/18 Z — not a 24-hour window ending at D 18 Z."""
    import castcheck.truth_instant as ti

    asked: list[tuple] = []

    def fake_obs(station, start, end):
        asked.append((start, end))
        idx = pd.date_range(start, end, freq="1h").floor("h") + pd.Timedelta(minutes=51)
        return pd.DataFrame({"time": idx, "temp_c": np.arange(len(idx), dtype=float),
                             "qc": [""] * len(idx)})

    monkeypatch.setattr(ti, "fetch_hourly_obs", fake_obs)
    out = ti.truth_instant_for_day([KNYC], date(2026, 8, 28))

    assert len(out) == 4
    assert list(out["valid_time"].dt.hour) == [0, 6, 12, 18]
    assert (out["valid_time"].dt.date == date(2026, 8, 28)).all()
    assert (out["source"] == SOURCE_NWS).all()
    # the fetch window is padded so the ±1 h suspect check has neighbours at both ends
    start, end = asked[0]
    assert start < pd.Timestamp("2026-08-28T00", tz="UTC")
    assert end > pd.Timestamp("2026-08-28T18", tz="UTC")


def test_a_quality_flagged_observation_is_not_used(monkeypatch):
    import castcheck.truth_instant as ti

    def fake_obs(station, start, end):
        return pd.DataFrame({
            "time": pd.to_datetime(["2026-08-28T11:51", "2026-08-28T12:15"], utc=True),
            "temp_c": [99.0, 20.0], "qc": ["X", ""],
        })

    monkeypatch.setattr(ti, "fetch_hourly_obs", fake_obs)
    row = ti._from_nws([KNYC], synoptic_times(date(2026, 8, 28), date(2026, 8, 28))).iloc[2]
    assert row.valid_time.hour == 12
    assert row.temp_c == pytest.approx(20.0)   # the "X" report is dropped before selection


def test_one_stations_outage_does_not_lose_the_batch(monkeypatch):
    import castcheck.truth_instant as ti

    def fake_obs(station, start, end):
        if station.id == "KMSP":
            raise RuntimeError("upstream is down")
        return pd.DataFrame({"time": pd.to_datetime(["2026-08-28T11:51"], utc=True),
                             "temp_c": [20.0], "qc": [""]})

    monkeypatch.setattr(ti, "fetch_hourly_obs", fake_obs)
    out = ti._from_nws([KNYC, KMSP], [at(12, "2026-08-28")])
    assert set(out["station_id"]) == {"KNYC", "KMSP"}
    assert out.set_index("station_id").loc["KMSP", "qc_flag"] == "no_report"
    assert out.set_index("station_id").loc["KNYC", "temp_c"] == pytest.approx(20.0)
