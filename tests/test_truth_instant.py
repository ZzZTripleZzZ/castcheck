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


# --------------------------------------------------------------------------- plausibility QC


"""The four cases below are real, taken from the 2024-2026 archive; each is a CLI report whose
first-final value contradicts every observation the station took that day, together with the
correction the NWS issued hours later."""

KDCA = Station(id="KDCA", name="Washington Reagan", cli_pil="CLIDCA", tz="America/New_York",
               std_offset_h=-5, lat=38.84833, lon=-77.03417, elev_m=4.0)
KLAX = Station(id="KLAX", name="Los Angeles Intl", cli_pil="CLILAX", tz="America/Los_Angeles",
               std_offset_h=-8, lat=33.93806, lon=-118.38889, elev_m=38.1)

KEWR = Station(id="KEWR", name="Newark Liberty", cli_pil="CLIEWR", tz="America/New_York",
               std_offset_h=-5, lat=40.6825, lon=-74.16944, elev_m=4.9)

#: station, climo date, (first-final tmax, tmin), (revised tmax, tmin), the day's four samples in °F,
#: and the values the check must publish. KDCA is the case that pins the rule down to *one variable
#: at a time*: its maximum is garbled and repaired, while its minimum (66 against a sampled 71) is
#: perfectly ordinary and must keep the first-final value even though a correction exists.
GARBLED_CLI = [
    (KLAX, date(2025, 2, 16), (69, 11), (69, 49), [52.0, 65.0, 65.0, 57.0], (69, 49)),  # "MINIMUM 11R"
    (KLAX, date(2024, 5, 5), (66, 17), (66, 53), [53.0, 62.0, 61.0, 62.0], (66, 53)),
    (KEWR, date(2026, 3, 4), (47, 24), (47, 37), [37.0, 39.0, 45.0, 46.0], (47, 37)),
    (KDCA, date(2025, 5, 18), (87, 66), (78, 68), [73.4, 71.0, 74.0, 74.0], (78, 66)),
]


def truth_row(station, climo_date, tmax_f, tmin_f, revised=(None, None), source="CLI",
              qc_flag="") -> dict:
    from castcheck.truth import f_to_c

    return {
        "station_id": station.id, "climo_date": climo_date, "source": source,
        "tmax_f": tmax_f, "tmin_f": tmin_f, "tmax_c": f_to_c(tmax_f), "tmin_c": f_to_c(tmin_f),
        "issuance_time": pd.Timestamp(climo_date, tz="UTC") + pd.Timedelta(days=1, hours=8),
        "is_final": True, "revised": revised != (None, None),
        "revised_tmax_f": revised[0], "revised_tmin_f": revised[1],
        "qc_flag": qc_flag, "product_id": "p", "schema_version": "0.3",
        "methodology_version": "0.3",
    }


def instants_for(station, climo_date, temps_f) -> pd.DataFrame:
    """A `truth_instant` frame holding the four samples of one station-day."""
    from castcheck.climo_day import common_sample_times

    times = common_sample_times(station, climo_date)
    return pd.DataFrame({
        "station_id": station.id,
        "valid_time": pd.to_datetime(times, utc=True),
        "temp_c": [(f - 32) * 5 / 9 for f in temps_f],
        "obs_time": pd.to_datetime(times, utc=True) - pd.Timedelta(minutes=9),
        "source": SOURCE_IEM, "n_reports": 1, "qc_flag": "",
        "schema_version": "0.3", "methodology_version": "0.3",
    })


@pytest.mark.parametrize("station,climo_date,first,rev,samples,expected", GARBLED_CLI,
                         ids=[f"{s.id}-{d}" for s, d, *_ in GARBLED_CLI])
def test_a_garbled_first_final_is_replaced_by_its_correction(station, climo_date, first, rev,
                                                             samples, expected):
    from castcheck.truth import QC_IMPLAUSIBLE, QC_REVISED_USED, plausibility_qc

    daily = pd.DataFrame([truth_row(station, climo_date, *first, revised=rev)])
    out = plausibility_qc(daily, instants_for(station, climo_date, samples), [station])
    row = out.iloc[0]

    assert (row.tmax_f, row.tmin_f) == expected, "the correction must become the published value"
    assert QC_IMPLAUSIBLE in row.qc_flag and QC_REVISED_USED in row.qc_flag
    assert row.methodology_version == "0.3.1"
    # °C mirrors the repaired °F, not the discarded one
    assert row.tmax_c == pytest.approx((expected[0] - 32) * 5 / 9, abs=1e-6)
    assert row.tmin_c == pytest.approx((expected[1] - 32) * 5 / 9, abs=1e-6)


def test_the_check_is_idempotent():
    from castcheck.truth import plausibility_qc

    station, climo_date, first, rev, samples, _ = GARBLED_CLI[0]
    instants = instants_for(station, climo_date, samples)
    once = plausibility_qc(pd.DataFrame([truth_row(station, climo_date, *first, revised=rev)]),
                           instants, [station])
    twice = plausibility_qc(once, instants, [station])
    assert (twice.iloc[0].tmax_f, twice.iloc[0].tmin_f) == rev
    assert twice.iloc[0].qc_flag == once.iloc[0].qc_flag


def test_an_ordinary_day_is_left_alone():
    """The daily extremes normally sit outside the samples; that is weather, not an error."""
    from castcheck.truth import plausibility_qc

    daily = pd.DataFrame([truth_row(KDCA, date(2025, 5, 18), 79, 66)])
    out = plausibility_qc(daily, instants_for(KDCA, date(2025, 5, 18), [73.4, 71.0, 74.0, 74.0]), [KDCA])
    assert (out.iloc[0].tmax_f, out.iloc[0].tmin_f) == (79, 66)
    assert out.iloc[0].qc_flag == ""


def test_a_large_excursion_with_nothing_to_corroborate_it_is_kept():
    """KOKC 2024-02-27: a 25 °F drop below the samples behind a winter front — real, and the
    hardest kind of day to forecast. Dropping it would bias the scores towards easy days."""
    from castcheck.truth import plausibility_qc

    daily = pd.DataFrame([truth_row(KDCA, date(2025, 1, 10), 45, 20)])
    out = plausibility_qc(daily, instants_for(KDCA, date(2025, 1, 10), [45.0, 44.0, 43.0, 44.0]), [KDCA])
    assert (out.iloc[0].tmax_f, out.iloc[0].tmin_f) == (45, 20)
    assert out.iloc[0].qc_flag == ""


def test_a_physically_impossible_value_with_no_alternative_is_dropped():
    """KMSY 2025-01-15: CLI reported a maximum of 51 °F on a day whose 17:53Z METAR read 58 °F."""
    from castcheck.truth import QC_DROPPED, plausibility_qc

    daily = pd.DataFrame([truth_row(KDCA, date(2025, 1, 15), 51, 50)])
    out = plausibility_qc(daily, instants_for(KDCA, date(2025, 1, 15), [51.0, 50.0, 58.0, 53.0]), [KDCA])
    row = out.iloc[0]
    assert pd.isna(row.tmax_f) and pd.isna(row.tmax_c)
    assert row.tmin_f == 50, "only the offending variable is dropped"
    assert QC_DROPPED in row.qc_flag


def test_cf6_is_used_when_the_correction_is_also_wrong():
    from castcheck.truth import QC_CF6_USED, plausibility_qc

    d = date(2025, 1, 15)
    daily = pd.DataFrame([truth_row(KDCA, d, 51, 50, revised=(52, 50)),      # correction still too low
                          truth_row(KDCA, d, 58, 50, source="CF6")])
    out = plausibility_qc(daily, instants_for(KDCA, d, [51.0, 50.0, 58.0, 53.0]), [KDCA])
    cli = out[out.source == "CLI"].iloc[0]
    assert cli.tmax_f == 58 and QC_CF6_USED in cli.qc_flag


def test_a_day_without_four_clean_samples_is_never_judged():
    """Three samples would raise a false alarm whenever the missing one was the extreme."""
    from castcheck.truth import plausibility_qc, sampled_extremes_f

    station, climo_date, first, rev, samples, _ = GARBLED_CLI[0]
    daily = pd.DataFrame([truth_row(station, climo_date, *first, revised=rev)])

    partial = instants_for(station, climo_date, samples).iloc[:3]
    assert plausibility_qc(daily, partial, [station]).iloc[0].tmin_f == first[1]

    suspect = instants_for(station, climo_date, samples)
    suspect.loc[suspect.index[1], "qc_flag"] = "suspect"
    assert sampled_extremes_f(suspect, [station]) == {}
    assert plausibility_qc(daily, suspect, [station]).iloc[0].tmin_f == first[1]


def test_an_empty_instant_table_changes_nothing():
    from castcheck.truth import plausibility_qc

    station, climo_date, first, rev, *_ = GARBLED_CLI[0]
    daily = pd.DataFrame([truth_row(station, climo_date, *first, revised=rev)])
    out = plausibility_qc(daily, pd.DataFrame(columns=TRUTH_INSTANT_COLUMNS), [station])
    assert (out.iloc[0].tmax_f, out.iloc[0].tmin_f) == first


def test_overwrite_truth_replaces_rows_that_upsert_would_refuse_to_change(data_dir):
    """`upsert_truth` keeps the earliest issuance; a re-run QC row has the *same* issuance time, so
    only an overwriting writer can land the correction."""
    from castcheck.store import overwrite_truth, read_truth, upsert_truth

    station, climo_date, first, rev, *_ = GARBLED_CLI[0]
    original = pd.DataFrame([truth_row(station, climo_date, *first, revised=rev)])
    upsert_truth(original)

    fixed = original.copy()
    fixed.loc[0, ["tmin_f", "qc_flag"]] = [rev[1], "cli_implausible"]
    upsert_truth(fixed)
    assert read_truth().iloc[0].tmin_f == first[1]      # first-final refuses it, by design

    overwrite_truth(fixed)
    stored = read_truth().iloc[0]
    assert stored.tmin_f == rev[1] and stored.qc_flag == "cli_implausible"


KATL = Station(id="KATL", name="Atlanta Hartsfield", cli_pil="CLIATL", tz="America/New_York",
               std_offset_h=-5, lat=33.64028, lon=-84.42694, elev_m=313.0)
KOKC = Station(id="KOKC", name="Oklahoma City", cli_pil="CLIOKC", tz="America/Chicago",
               std_offset_h=-6, lat=35.38861, lon=-97.60028, elev_m=394.1)


def test_an_excursion_past_the_observed_envelope_is_dropped_even_without_a_replacement():
    """KATL 2026-04-14: minimum 32 °F against samples of 65/63/82/80. The correction that day
    touched only the maximum, so `revised_tmin_f` is absent and there is no CF6 row — nothing can
    replace the value, and 31 °F is further outside the samples than any real excursion in three
    years of this archive."""
    from castcheck.truth import QC_DROPPED, QC_IMPLAUSIBLE, plausibility_qc

    d = date(2026, 4, 14)
    daily = pd.DataFrame([truth_row(KATL, d, 86, 32, revised=(86, None))])
    row = plausibility_qc(daily, instants_for(KATL, d, [65.0, 63.0, 82.0, 80.0]), [KATL]).iloc[0]

    assert pd.isna(row.tmin_f) and pd.isna(row.tmin_c)
    assert row.tmax_f == 86, "the maximum is consistent with the samples and is left alone"
    assert QC_IMPLAUSIBLE in row.qc_flag and QC_DROPPED in row.qc_flag


def test_the_widest_real_excursion_in_the_archive_is_still_kept():
    """KOKC 2024-02-27: minimum 37 °F, 25 °F below the day's lowest sample, because the front
    arrived after the last one. This is the value the envelope is drawn at — it must survive, or
    the bound is cutting into real weather instead of bounding it."""
    from castcheck.truth import plausibility_qc

    d = date(2024, 2, 27)
    daily = pd.DataFrame([truth_row(KOKC, d, 80, 37)])
    row = plausibility_qc(daily, instants_for(KOKC, d, [62.0, 64.0, 73.0, 74.0]), [KOKC]).iloc[0]

    assert (row.tmax_f, row.tmin_f) == (80, 37)
    assert row.qc_flag == ""


def test_the_envelope_bound_is_compared_on_the_whole_degree_lattice():
    """A 25 °F excursion arrives as 24.999999 after the °C round trip and must not be dropped."""
    from castcheck.truth import PLAUSIBILITY_REVIEW_F, plausibility_qc

    d = date(2024, 2, 27)
    samples = [62.0, 64.0, 73.0, 74.0]
    at_bound = int(min(samples) - PLAUSIBILITY_REVIEW_F)          # 37 °F, exactly on the bound
    just_past = at_bound - 1                                       # 36 °F, one step outside it

    keep = plausibility_qc(pd.DataFrame([truth_row(KOKC, d, 80, at_bound)]),
                           instants_for(KOKC, d, samples), [KOKC]).iloc[0]
    drop = plausibility_qc(pd.DataFrame([truth_row(KOKC, d, 80, just_past)]),
                           instants_for(KOKC, d, samples), [KOKC]).iloc[0]

    assert keep.tmin_f == at_bound
    assert pd.isna(drop.tmin_f)
