from datetime import UTC, date, datetime

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
