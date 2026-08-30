"""Climatological-day arithmetic (METHODOLOGY §2).

A climatological day is midnight-to-midnight in *local standard time* (DST ignored), matching the NWS
CLI product. All datetimes handled here are timezone-aware UTC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from .config import Station

COMMON_SAMPLE_HOURS_UTC = (0, 6, 12, 18)


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def day_bounds_utc(station: Station, climo_date: date) -> tuple[datetime, datetime]:
    """Half-open UTC interval [start, end) of the station's climatological day."""
    local_midnight = datetime(climo_date.year, climo_date.month, climo_date.day, tzinfo=UTC)
    start = local_midnight - timedelta(hours=station.std_offset_h)
    return start, start + timedelta(hours=24)


def common_sample_times(station: Station, climo_date: date) -> list[datetime]:
    """The four 00/06/12/18 UTC instants that fall inside the climatological day."""
    start, end = day_bounds_utc(station, climo_date)
    first = start.replace(minute=0, second=0, microsecond=0)
    while first.hour % 6 != 0 or first < start:
        first += timedelta(hours=1)
    out = []
    t = first
    while t < end:
        out.append(t)
        t += timedelta(hours=6)
    assert len(out) == 4, (station.id, climo_date, out)
    return out


def lead_day(init_time: datetime, climo_date: date) -> int:
    """climo_date minus the UTC date of the initialization."""
    return (climo_date - _utc(init_time).date()).days


def climo_dates_for_run(station: Station, init_time: datetime, max_h: int, allow_f000: bool = False) -> list[date]:
    """Climatological dates whose four common samples are all within (init, init+max_h].

    f000 is excluded by default because AIWP files carry a fill value at f000 and IFS/GFS analysis
    fields are not forecasts.
    """
    init = _utc(init_time)
    horizon = init + timedelta(hours=max_h)
    out: list[date] = []
    d = init.date() - timedelta(days=1)
    while True:
        samples = common_sample_times(station, d)
        lo_ok = all((t > init) or (allow_f000 and t == init) for t in samples)
        hi_ok = all(t <= horizon for t in samples)
        if lo_ok and hi_ok:
            out.append(d)
        if samples[0] > horizon:
            break
        d += timedelta(days=1)
    return out


def lead_hours(init_time: datetime, valid_time: datetime) -> int:
    delta = _utc(valid_time) - _utc(init_time)
    h = delta.total_seconds() / 3600
    if h != int(h):
        raise ValueError("non-integer lead")
    return int(h)
