"""Tests for the truth package: CLI/CF6/obs parsing and the first-final policy.

Fixtures in ``tests/fixtures/`` are verbatim NWS products pulled from the IEM AFOS archive:

* ``cli_final_yesterday_knyc.txt``   CLINYC 2026-08-30 06:42Z — the first-final for 2026-08-29
* ``cli_intermediate_today_knyc.txt``CLINYC 2026-08-30 20:41Z — same-day "VALID TODAY AS OF"
* ``cli_valid_as_of_kmsp.txt``       CLIMSP 2026-08-29 21:31Z — "VALID AS OF" wording, TODAY block
* ``cli_corrected_knyc.txt``         CLINYC 2026-08-27 21:48Z CCA — "CLIMATE REPORT...CORRECTED"
* ``cli_missing_value_kmia.txt``     CLIMIA 2026-07-08 09:26Z — MINIMUM reported as ``MM``
* ``cli_first_final_kmia.txt`` / ``cli_correction_kmia.txt``  the real 2026-08-29 Miami pair where
  the first final said 90 °F and a report 46 minutes later said 85 °F (CF6 confirms 85)
* ``cf6_knyc_202608.txt``            CF6NYC 2026-08-30 09:10Z
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from castcheck.config import Station
from castcheck.sources.nws_cf6 import parse_cf6
from castcheck.sources.nws_cli import parse_cli, parse_issuance_time
from castcheck.sources.nws_obs import daily_extremes_from_obs
from castcheck.store import TRUTH_COLUMNS, TRUTH_KEY, _apply_first_final
from castcheck.truth import best_truth, build_truth_rows, f_to_c

FIXTURES = Path(__file__).parent / "fixtures"

KNYC = Station(id="KNYC", name="New York Central Park", cli_pil="CLINYC", tz="America/New_York",
               std_offset_h=-5, lat=40.78333, lon=-73.96667, elev_m=46.9)
KMIA = Station(id="KMIA", name="Miami Intl", cli_pil="CLIMIA", tz="America/New_York",
               std_offset_h=-5, lat=25.79056, lon=-80.31639, elev_m=3.0)


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- CLI parsing


def test_parse_final_yesterday_block():
    p = parse_cli(fixture("cli_final_yesterday_knyc.txt"))
    assert p is not None
    assert p["climo_date"] == date(2026, 8, 29)
    assert p["block"] == "YESTERDAY"
    assert p["is_final"] is True
    assert p["is_corrected"] is False
    assert (p["tmax_f"], p["tmin_f"]) == (76, 64)
    assert (p["tmax_time"], p["tmin_time"]) == ("3:55 PM", "7:19 AM")
    assert p["issuance_time"] == datetime(2026, 8, 30, 6, 42, tzinfo=UTC)
    assert p["station_hint"] == "CENTRAL PARK NY"
    assert p["pil"] == "CLINYC"
    assert p["office"] == "KOKX"


def test_parse_intermediate_report_is_not_final():
    """A same-day "VALID TODAY AS OF" report describes the day in progress; it is never truth."""
    p = parse_cli(fixture("cli_intermediate_today_knyc.txt"))
    assert p["block"] == "TODAY"
    assert p["is_final"] is False
    assert p["climo_date"] == date(2026, 8, 30)  # the day it is issued on, not the previous one
    assert (p["tmax_f"], p["tmin_f"]) == (78, 67)


def test_parse_valid_as_of_wording_still_reads_today_block():
    """Some offices write "VALID AS OF" without the word TODAY; the block header decides."""
    p = parse_cli(fixture("cli_valid_as_of_kmsp.txt"))
    assert p["block"] == "TODAY"
    assert p["is_final"] is False
    assert p["climo_date"] == date(2026, 8, 29)
    assert (p["tmax_f"], p["tmin_f"]) == (82, 67)


def test_parse_corrected_report():
    p = parse_cli(fixture("cli_corrected_knyc.txt"))
    assert p["is_corrected"] is True
    assert p["wmo_suffix"] == "CCA"
    assert p["climo_date"] == date(2026, 8, 27)
    assert (p["tmax_f"], p["tmin_f"]) == (77, 70)


def test_parse_missing_value_M():
    """``M``/``MM`` makes the variable missing for the day (METHODOLOGY §6)."""
    p = parse_cli(fixture("cli_missing_value_kmia.txt"))
    assert p["climo_date"] == date(2026, 7, 7)
    assert p["block"] == "YESTERDAY"
    assert p["tmax_f"] == 93
    assert p["tmin_f"] is None
    assert p["tmin_time"] is None
    assert p["tmax_time"] == "1:33 PM"


def test_parse_returns_none_for_non_cli_text():
    assert parse_cli("") is None
    assert parse_cli("AREA FORECAST DISCUSSION\nNOTHING TO SEE HERE") is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("cli_final_yesterday_knyc.txt", datetime(2026, 8, 30, 6, 42, tzinfo=UTC)),
        ("cli_corrected_knyc.txt", datetime(2026, 8, 27, 21, 48, tzinfo=UTC)),
        ("cli_valid_as_of_kmsp.txt", datetime(2026, 8, 29, 21, 31, tzinfo=UTC)),
    ],
)
def test_issuance_time_from_local_header_line(name, expected):
    """The "242 AM EDT SUN AUG 30 2026" line reconstructs the UTC issuance without an API call."""
    assert parse_issuance_time(fixture(name)) == expected


# --------------------------------------------------------------------------- CF6 parsing


def test_parse_cf6_month_table():
    df = parse_cf6(fixture("cf6_knyc_202608.txt"))
    assert len(df) == 29  # issued 30 August: the month is only complete through the 29th
    assert df["climo_date"].iloc[0] == date(2026, 8, 1)
    row = df[df["climo_date"] == date(2026, 8, 29)].iloc[0]
    assert (row["tmax_f"], row["tmin_f"]) == (76, 64)
    assert row["issuance_time"] == pd.Timestamp("2026-08-30T09:10Z")
    assert row["product_id"] == "202608300910-KOKX-CXUS51-CF6NYC"
    assert df["tmax_f"].notna().all() and df["tmin_f"].notna().all()


def test_parse_cf6_rejects_other_products():
    assert parse_cf6(fixture("cli_final_yesterday_knyc.txt")).empty


# --------------------------------------------------------------------------- obs


def _obs(pairs) -> pd.DataFrame:
    return pd.DataFrame(
        {"time": pd.to_datetime([t for t, _, _ in pairs], utc=True),
         "temp_c": [c for _, c, _ in pairs],
         "qc": [q for _, _, q in pairs]}
    )


def test_daily_extremes_respect_lst_day_bounds_and_qc():
    """The LST day for a −5 h station is [05Z, 05Z+24h); values outside and bad QC are dropped."""
    obs = _obs([
        ("2026-08-29T04:53Z", 40.0, "V"),   # before the day starts
        ("2026-08-29T11:53Z", 17.8, "V"),
        ("2026-08-29T19:53Z", 24.4, "V"),
        ("2026-08-29T20:53Z", 99.9, "X"),   # failed QC
        ("2026-08-30T05:53Z", -40.0, "V"),  # after the day ends
    ])
    tmax_f, tmin_f = daily_extremes_from_obs(obs, KNYC, date(2026, 8, 29))
    assert round(tmax_f, 1) == 75.9
    assert round(tmin_f, 1) == 64.0


def test_daily_extremes_prefer_high_resolution_reports():
    """Whole-°C 5-minute values must not beat the 0.1 °C routine METAR readings."""
    obs = _obs([
        ("2026-08-29T19:50Z", 30.0, "V"),   # 5-minute feed, rounded to whole °C
        ("2026-08-29T19:53Z", 29.4, "V"),   # routine METAR
        ("2026-08-29T20:50Z", 29.0, "V"),
        ("2026-08-29T20:53Z", 28.9, "V"),
    ])
    tmax_f, _ = daily_extremes_from_obs(obs, KNYC, date(2026, 8, 29))
    assert round(tmax_f, 1) == 84.9  # 29.4 °C, not 30 °C


def test_daily_extremes_empty():
    assert daily_extremes_from_obs(pd.DataFrame(), KNYC, date(2026, 8, 29)) == (None, None)


# --------------------------------------------------------------------------- row assembly


def test_build_truth_rows_one_row_per_source():
    cli = parse_cli(fixture("cli_final_yesterday_knyc.txt"))
    cf6 = parse_cf6(fixture("cf6_knyc_202608.txt"))
    df = build_truth_rows(KNYC, date(2026, 8, 29), cli=cli, cf6=cf6, obs=(75.0, 64.0))

    assert list(df.columns) == TRUTH_COLUMNS
    assert sorted(df["source"]) == ["CF6", "CLI", "OBS"]
    cli_row = df[df["source"] == "CLI"].iloc[0]
    assert (cli_row["tmax_f"], cli_row["tmin_f"]) == (76, 64)
    assert cli_row["tmax_c"] == pytest.approx(f_to_c(76))
    assert cli_row["is_final"] and not cli_row["revised"]
    assert cli_row["qc_flag"] == ""  # 1 °F apart, below the 2 °F threshold
    assert df[df["source"] == "OBS"].iloc[0]["qc_flag"] == "obs_fallback"


def test_build_truth_rows_flags_obs_disagreement():
    cli = parse_cli(fixture("cli_final_yesterday_knyc.txt"))
    df = build_truth_rows(KNYC, date(2026, 8, 29), cli=cli, obs=(70.0, 64.0))
    assert df[df["source"] == "CLI"].iloc[0]["qc_flag"] == "obs_diff_gt2f"


def test_build_truth_rows_flags_missing_cli_value():
    cli = parse_cli(fixture("cli_missing_value_kmia.txt"))
    df = build_truth_rows(KMIA, date(2026, 7, 7), cli=cli)
    row = df.iloc[0]
    assert row["tmax_f"] == 93
    assert pd.isna(row["tmin_f"])
    assert pd.isna(row["tmin_c"])
    assert row["qc_flag"] == "cli_missing_value"


def test_build_truth_rows_records_later_correction():
    """The first final is published; a later differing issuance only fills revised_*."""
    first = parse_cli(fixture("cli_first_final_kmia.txt"))
    later = parse_cli(fixture("cli_correction_kmia.txt"))
    first["later_versions"] = [later]
    df = build_truth_rows(KMIA, date(2026, 8, 29), cli=first)
    row = df.iloc[0]
    assert (row["tmax_f"], row["tmin_f"]) == (90, 76)
    assert row["revised"] is True or bool(row["revised"])
    assert row["revised_tmax_f"] == 85
    assert row["revised_tmin_f"] == 76


def test_best_truth_source_priority():
    cli = parse_cli(fixture("cli_final_yesterday_knyc.txt"))
    cf6 = parse_cf6(fixture("cf6_knyc_202608.txt"))
    df = build_truth_rows(KNYC, date(2026, 8, 29), cli=cli, cf6=cf6, obs=(75.0, 64.0))
    best = best_truth(df)
    assert len(best) == 1
    assert best.iloc[0]["source"] == "CLI"


# --------------------------------------------------------------------------- first-final policy


def _truth_row(**over) -> dict:
    base = dict(
        station_id="KNYC", climo_date=date(2026, 8, 29), source="CLI", tmax_f=76, tmin_f=64,
        tmax_c=f_to_c(76), tmin_c=f_to_c(64),
        issuance_time=pd.Timestamp("2026-08-30T06:42Z"), is_final=True, revised=False,
        revised_tmax_f=None, revised_tmin_f=None, qc_flag="", product_id="first",
        schema_version="0.1", methodology_version="0.1",
    )
    base.update(over)
    return base


def test_first_final_keeps_earliest_issuance_and_merges_corrections():
    later = _truth_row(tmax_f=85, issuance_time=pd.Timestamp("2026-08-30T09:10Z"),
                       revised=True, revised_tmax_f=85, revised_tmin_f=64,
                       qc_flag="obs_diff_gt2f", product_id="later")
    merged = _apply_first_final(pd.DataFrame([_truth_row(), later], columns=TRUTH_COLUMNS))
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["tmax_f"] == 76             # published value unchanged
    assert row["product_id"] == "first"
    assert bool(row["revised"]) is True    # the correction is recorded
    assert row["revised_tmax_f"] == 85
    assert row["qc_flag"] == "obs_diff_gt2f"


def test_first_final_is_order_independent():
    a = _truth_row()
    b = _truth_row(tmax_f=85, issuance_time=pd.Timestamp("2026-08-30T09:10Z"), product_id="later")
    fwd = _apply_first_final(pd.DataFrame([a, b], columns=TRUTH_COLUMNS))
    rev = _apply_first_final(pd.DataFrame([b, a], columns=TRUTH_COLUMNS))
    assert fwd.iloc[0]["tmax_f"] == rev.iloc[0]["tmax_f"] == 76


def test_first_final_keeps_sources_apart():
    rows = [_truth_row(), _truth_row(source="CF6", tmax_f=85, product_id="cf6")]
    merged = _apply_first_final(pd.DataFrame(rows, columns=TRUTH_COLUMNS))
    assert len(merged) == 2
    assert set(merged["source"]) == {"CLI", "CF6"}


def test_first_final_survives_mixed_present_and_missing_corrections():
    """Regression: `truth-backfill` over a long range crashed with ``KeyError: np.int64(N)``.

    A year of station-days contains both corrected reports (``revised_tmax_f`` present) and reports
    that are merely flagged as revised (``revised_tmax_f`` missing). Writing that mixture into the
    nullable ``Int16`` column through ``DataFrame.loc`` tripped a pandas bug that indexes the value
    Series positionally while it carries the ``(station_id, climo_date, source)`` MultiIndex.
    """
    rows = []
    for i in range(12):
        day = date(2026, 8, 1) + timedelta(days=i)
        rows.append(_truth_row(climo_date=day))
        # every other day carries an actual corrected value; the rest are flagged only
        rows.append(_truth_row(
            climo_date=day, issuance_time=pd.Timestamp("2026-08-30T09:10Z"), revised=True,
            revised_tmax_f=(80 + i) if i % 2 == 0 else None,
            revised_tmin_f=(60 + i) if i % 2 == 0 else None, product_id="later",
        ))
    frame = pd.DataFrame(rows, columns=TRUTH_COLUMNS)
    frame["revised_tmax_f"] = frame["revised_tmax_f"].astype("Int16")
    frame["revised_tmin_f"] = frame["revised_tmin_f"].astype("Int16")

    merged = _apply_first_final(frame)

    assert len(merged) == 12
    assert merged["revised"].all()
    assert str(merged["revised_tmax_f"].dtype) == "Int16"
    assert merged["revised_tmax_f"].notna().sum() == 6
    assert (merged["tmax_f"] == 76).all()          # published values never move
    by_day = merged.set_index("climo_date")["revised_tmax_f"]
    assert by_day[date(2026, 8, 1)] == 80
    assert pd.isna(by_day[date(2026, 8, 2)])


def test_upsert_truth_is_idempotent(tmp_path, monkeypatch):
    from castcheck import store

    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    df = pd.DataFrame([_truth_row(), _truth_row(source="OBS", tmax_f=75, qc_flag="obs_fallback")],
                      columns=TRUTH_COLUMNS)

    assert store.upsert_truth(df) == {"truth_daily/year=2026.parquet": 2}
    once = store.read_truth([2026])
    assert store.upsert_truth(df) == {"truth_daily/year=2026.parquet": 2}
    twice = store.read_truth([2026])

    pd.testing.assert_frame_equal(once, twice)
    assert len(twice) == 2
    assert set(twice.columns) == set(TRUTH_COLUMNS)
    assert twice.duplicated(subset=TRUTH_KEY).sum() == 0


def test_upsert_truth_second_run_cannot_change_published_value(tmp_path, monkeypatch):
    from castcheck import store

    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.upsert_truth(pd.DataFrame([_truth_row()], columns=TRUTH_COLUMNS))
    store.upsert_truth(pd.DataFrame([
        _truth_row(tmax_f=85, issuance_time=pd.Timestamp("2026-08-30T09:10Z"),
                   revised=True, revised_tmax_f=85, revised_tmin_f=64, product_id="later")
    ], columns=TRUTH_COLUMNS))

    stored = store.read_truth([2026])
    assert len(stored) == 1
    assert stored.iloc[0]["tmax_f"] == 76
    assert stored.iloc[0]["revised_tmax_f"] == 85
    assert bool(stored.iloc[0]["revised"]) is True


# --------------------------------------------------------------------------- network


@pytest.mark.network
def test_live_cli_first_final_matches_cf6():
    from castcheck.sources.nws_cf6 import fetch_cf6
    from castcheck.sources.nws_cli import fetch_cli_day

    day = date(2026, 8, 29)
    cli = fetch_cli_day(KNYC, day)
    assert cli is not None and cli["block"] == "YESTERDAY"
    cf6 = fetch_cf6(KNYC, day.year, day.month).set_index("climo_date")
    assert (cli["tmax_f"], cli["tmin_f"]) == (cf6.loc[day, "tmax_f"], cf6.loc[day, "tmin_f"])
