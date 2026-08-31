from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from castcheck.climo_day import climo_dates_for_run, common_sample_times, day_bounds_utc, lead_day
from castcheck.config import Station, standard_offset_hours
from castcheck.grid import bilinear, nearest


def st(tz: str, off: int) -> Station:
    return Station(id="TST", name="t", cli_pil="CLITST", tz=tz, std_offset_h=off, lat=40.0, lon=-74.0, elev_m=10.0)


@pytest.mark.parametrize(
    "tz,off",
    [("America/New_York", -5), ("America/Chicago", -6), ("America/Denver", -7), ("America/Phoenix", -7), ("America/Los_Angeles", -8)],
)
def test_standard_offsets(tz, off):
    assert standard_offset_hours(tz) == off


def test_day_bounds_and_samples_est_dst_date():
    s = st("America/New_York", -5)
    start, end = day_bounds_utc(s, date(2026, 7, 4))  # DST date, but LST used
    assert start == datetime(2026, 7, 4, 5, tzinfo=UTC)
    assert end == datetime(2026, 7, 5, 5, tzinfo=UTC)
    samples = common_sample_times(s, date(2026, 7, 4))
    assert [t.hour for t in samples] == [6, 12, 18, 0]
    assert samples[-1].date() == date(2026, 7, 5)


def test_samples_pst():
    s = st("America/Los_Angeles", -8)
    samples = common_sample_times(s, date(2026, 1, 10))
    assert [t.hour for t in samples] == [12, 18, 0, 6]


def test_lead_day_and_run_coverage():
    s = st("America/New_York", -5)
    init = datetime(2026, 8, 30, 0, tzinfo=UTC)
    assert lead_day(init, date(2026, 8, 31)) == 1
    days = climo_dates_for_run(s, init, 240)
    assert days[0] == date(2026, 8, 30)  # lead day 0: samples f6..f24
    assert lead_day(init, days[-1]) == 9
    init12 = datetime(2026, 8, 30, 12, tzinfo=UTC)
    days12 = climo_dates_for_run(s, init12, 240)
    assert days12[0] == date(2026, 8, 31)  # 06Z sample of day 0 precedes a 12Z init


@pytest.mark.parametrize("climo_date", [date(2026, 3, 7), date(2026, 3, 8), date(2026, 3, 9),
                                        date(2026, 10, 31), date(2026, 11, 1), date(2026, 11, 2)])
@pytest.mark.parametrize("tz,off", [("America/New_York", -5), ("America/Chicago", -6),
                                    ("America/Denver", -7), ("America/Phoenix", -7),
                                    ("America/Los_Angeles", -8)])
def test_climatological_day_is_unaffected_by_dst_transitions(tz, off, climo_date):
    """METHODOLOGY §2.1: the day is LST midnight-to-midnight, so the US DST switch dates
    (2026-03-08 spring forward, 2026-11-01 fall back) are ordinary 24 h days, and Phoenix — which
    never observes DST — behaves exactly like the rest of Mountain time."""
    s = st(tz, off)
    start, end = day_bounds_utc(s, climo_date)
    assert end - start == timedelta(hours=24)
    assert start.hour == (-off) % 24
    samples = common_sample_times(s, climo_date)
    assert len(samples) == 4
    assert all(start <= t < end for t in samples)
    assert {t.hour for t in samples} == {0, 6, 12, 18}


@pytest.mark.parametrize("off,expect_last_lead", [(-5, 9), (-6, 9), (-7, 8), (-8, 8)])
def test_00z_run_reaches_lead_9_only_for_eastern_stations(off, expect_last_lead):
    """A 240 h horizon from 00 UTC ends before the 06 UTC sample that closes lead day 9 at a −7/−8 h
    station, so the ALL row at lead 9 rests on fewer stations than at lead 8 (METHODOLOGY §2.5)."""
    s = st("America/New_York", off)
    init = datetime(2026, 8, 30, 0, tzinfo=UTC)
    days = climo_dates_for_run(s, init, 240)
    assert lead_day(init, days[0]) == 0
    assert lead_day(init, days[-1]) == expect_last_lead


def test_extraction_is_identical_in_both_longitude_conventions():
    """METHODOLOGY §2.6: the same physical field must give the same station value whether it is
    stored −180..179.75 (ECMWF) or 0..359.75 (GFS/AIWP), on a latitude-descending grid."""
    lats = np.linspace(90.0, -90.0, 121)  # 1.5° for speed; descending like every source
    lon_e = np.arange(-180.0, 180.0, 1.5)
    lon_g = np.arange(0.0, 360.0, 1.5)

    def field(lo):
        la = np.deg2rad(lats)[:, None]
        x = np.deg2rad(lo)[None, :]
        return 288.0 + 20.0 * np.sin(la) + 3.0 * np.cos(3 * x) * np.cos(la)

    fe, fg = field(lon_e), field(lon_g)
    for lat, lon in [(40.78, -73.97), (47.45, -122.31), (33.43, -112.01), (0.0, 0.0),
                     (51.5, -0.12), (-33.9, 151.2), (64.0, 179.9), (64.0, -179.9)]:
        assert bilinear(fe, lats, lon_e, lat, lon) == pytest.approx(
            bilinear(fg, lats, lon_g, lat, lon), abs=1e-9
        )
        assert nearest(fe, lats, lon_e, lat, lon) == pytest.approx(
            nearest(fg, lats, lon_g, lat, lon), abs=1e-9
        )


def test_bilinear_wraps_across_the_prime_and_date_meridians():
    lats = np.array([1.0, 0.0, -1.0])          # descending
    lons = np.array([0.0, 120.0, 240.0])       # coarse 0..360 grid: the last gap wraps to 0
    field = np.tile(np.array([0.0, 12.0, 24.0]), (3, 1))
    # halfway between the 240° node (24) and the wrapped 0°/360° node (0)
    assert bilinear(field, lats, lons, 0.0, 300.0) == pytest.approx(12.0)
    assert bilinear(field, lats, lons, 0.0, -60.0) == pytest.approx(12.0)  # same point, signed
    assert bilinear(field, lats, lons, 0.0, 359.9) == pytest.approx(24.0 * (0.1 / 120.0), abs=1e-9)


def test_bilinear_and_nearest_conventions():
    lats = np.array([50.0, 49.75, 49.5])  # descending like ECMWF
    lons = np.array([0.0, 0.25, 0.5, 0.75])
    field = lats[:, None] * 0 + lons[None, :] * 4 + (50.0 - lats)[:, None] * 8  # planar: 4*lon + 8*(50-lat)
    assert bilinear(field, lats, lons, 49.875, 0.375) == pytest.approx(4 * 0.375 + 8 * 0.125)
    assert nearest(field, lats, lons, 49.80, 0.30) == pytest.approx(4 * 0.25 + 8 * 0.25)
    # 0..360 field queried with negative longitude
    lons360 = np.array([359.5, 359.75, 0.0, 0.25]) % 360  # unsorted wrap not expected; use sorted grid
    lons360 = np.array([0.0, 0.25, 359.5, 359.75])
    f2 = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (3, 1))
    assert nearest(f2, lats, lons360, 49.9, -0.25) == pytest.approx(4.0)
