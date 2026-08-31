"""Observed 2 m temperature at the four synoptic instants — the ``truth_instant`` table (DESIGN §10.1).

Why this table exists
---------------------
Until v0.2 the headline metric compared each model's *sampled* daily extreme (the max of its four
6-hourly values) against the NWS CLI daily extreme, which is the true max of a continuous trace.
The external review (§A2) showed that the resulting penalty is not a constant offset: its size
depends on how much diurnal amplitude each model carries, and that amplitude is exactly what the
site is trying to measure. Scoring the models at the instants they are actually sampled at removes
the confound, and that needs an observation at those same instants.

How an instant is resolved
--------------------------
For each of 00/06/12/18 UTC the routine METAR nearest the hour within :data:`WINDOW_MIN` minutes is
used. Reports at the scheduled minutes (:data:`PREFERRED_MINUTES`, i.e. :51–:56 — the window in
which US ASOS sites transmit the hourly observation) win over anything else in the window, so a
SPECI two minutes from the hour never displaces the scheduled report it was issued alongside; among
equally preferred candidates the closest to the hour wins.

``qc_flag`` records why a value is missing or doubtful, and the flags are deliberately distinct:

``no_report``     nothing usable anywhere near the instant (station down, archive gap);
``gap_gt35min``   the station has usable reports around the instant but none inside ±35 min;
``suspect``       a value exists but differs by more than :data:`SUSPECT_JUMP_C` from the report an
                  hour before or after — a real air temperature does not step 8 °C in an hour
                  outside a frontal passage, and a sensor sticking or a decode error does.

Flagged rows are kept, never dropped: the flag is published beside the value (METHODOLOGY §6).

Two sources feed the table. :func:`truth_instant_for_range` reads the IEM ASOS archive
(``source="ASOS_IEM"``), which lags real time by a few hours to a day; :func:`truth_instant_recent`
reads api.weather.gov (``source="NWS_API"``) and covers the last week, so the daily pipeline can
score yesterday before the archive catches up. :func:`castcheck.store.upsert_truth_instant` prefers
the archive when both exist.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION
from .config import Station, load_stations
from .sources.iem_asos import fetch_asos
from .sources.nws_obs import BAD_QC, fetch_hourly_obs
from .store import TRUTH_INSTANT_COLUMNS

log = logging.getLogger(__name__)

#: the UTC hours every model in models.yaml is sampled at (step_h = 6)
SYNOPTIC_HOURS = (0, 6, 12, 18)

#: half-width of the window a report may be taken from, minutes
WINDOW_MIN = 35

#: minutes at which US ASOS sites transmit the scheduled hourly METAR
PREFERRED_MINUTES = frozenset(range(51, 57))

#: |ΔT| against the report one hour away, above which the value is flagged ``suspect`` (°C)
SUSPECT_JUMP_C = 8.0

#: how far out to look before concluding a station reported nothing at all, minutes
NEAR_SEARCH_MIN = 180

SOURCE_IEM = "ASOS_IEM"
SOURCE_NWS = "NWS_API"

_WINDOW_NS = WINDOW_MIN * 60 * 1_000_000_000
_NEAR_NS = NEAR_SEARCH_MIN * 60 * 1_000_000_000
_HOUR_NS = 3600 * 1_000_000_000


def empty_truth_instant() -> pd.DataFrame:
    df = pd.DataFrame(columns=TRUTH_INSTANT_COLUMNS)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)
    df["temp_c"] = df["temp_c"].astype("float32")
    df["n_reports"] = df["n_reports"].astype("int8")
    return df


# --------------------------------------------------------------------------- selection


class _Series:
    """A station's reports as sorted numpy arrays, so one instant costs two ``searchsorted`` calls.

    A backfill resolves ~3800 instants per station; slicing a DataFrame that many times is a minute
    per station of pure pandas overhead, and the arrays make it milliseconds.
    """

    def __init__(self, reports: pd.DataFrame):
        if reports is None or len(reports) == 0:
            self.t = np.empty(0, dtype="int64")
            self.v = np.empty(0, dtype="float64")
            self.minute = np.empty(0, dtype="int64")
            self.ut = np.empty(0, dtype="int64")
            self.uv = np.empty(0, dtype="float64")
            self.uminute = np.empty(0, dtype="int64")
            return
        df = reports.copy()
        df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)
        df = df.sort_values("obs_time")
        # to_numpy("datetime64[ns]") pins the unit: pandas 2 keeps parsed timestamps as
        # ``datetime64[us]``, whose astype("int64") is microseconds, while Timestamp.value is
        # always nanoseconds — mixing the two silently puts every report 1000x too far from its
        # instant, and every instant then reads as "no report".
        self.t = df["obs_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        self.v = pd.to_numeric(df["temp_c"], errors="coerce").to_numpy(dtype="float64")
        self.minute = df["obs_time"].dt.minute.to_numpy()
        usable = ~np.isnan(self.v)
        self.ut, self.uv, self.uminute = self.t[usable], self.v[usable], self.minute[usable]

    def n_in_window(self, t_ns: int) -> int:
        lo = np.searchsorted(self.t, t_ns - _WINDOW_NS, side="left")
        hi = np.searchsorted(self.t, t_ns + _WINDOW_NS, side="right")
        return int(hi - lo)

    def pick(self, t_ns: int) -> tuple[float, int] | None:
        """The usable report to represent instant ``t_ns``: ``(temp_c, obs_time_ns)`` or ``None``."""
        lo = np.searchsorted(self.ut, t_ns - _WINDOW_NS, side="left")
        hi = np.searchsorted(self.ut, t_ns + _WINDOW_NS, side="right")
        if hi <= lo:
            return None
        dt = np.abs(self.ut[lo:hi] - t_ns)
        preferred = np.isin(self.uminute[lo:hi], list(PREFERRED_MINUTES))
        # lexsort: scheduled reports first, then closest to the hour, then earliest
        order = np.lexsort((self.ut[lo:hi], dt, ~preferred))
        j = lo + int(order[0])
        return float(self.uv[j]), int(self.ut[j])

    def has_usable_near(self, t_ns: int) -> bool:
        lo = np.searchsorted(self.ut, t_ns - _NEAR_NS, side="left")
        hi = np.searchsorted(self.ut, t_ns + _NEAR_NS, side="right")
        return bool(hi > lo)


def instant_from_reports(
    reports: pd.DataFrame, station: Station, valid_times, *, source: str = SOURCE_IEM,
) -> pd.DataFrame:
    """Reduce one station's report series to one ``truth_instant`` row per requested instant.

    ``reports`` needs ``obs_time`` (UTC) and ``temp_c`` (°C, ``NaN`` allowed) — the shape returned
    by :func:`castcheck.sources.iem_asos.fetch_asos`. Every instant in ``valid_times`` gets a row,
    including the ones with no observation, because an absent row and an absent *value* are
    different statements and only the second one is a fact about the station.
    """
    times = pd.DatetimeIndex(pd.to_datetime(list(valid_times), utc=True)).sort_values().unique()
    if len(times) == 0:
        return empty_truth_instant()
    s = _Series(reports)

    rows = []
    for ts in times:
        t_ns = int(pd.Timestamp(ts).value)
        n = s.n_in_window(t_ns)
        chosen = s.pick(t_ns)
        if chosen is None:
            temp, obs_ns = float("nan"), None
            flag = "gap_gt35min" if s.has_usable_near(t_ns) else "no_report"
        else:
            temp, obs_ns = chosen
            flag = ""
            for other in (s.pick(t_ns - _HOUR_NS), s.pick(t_ns + _HOUR_NS)):
                if other is not None and abs(other[0] - temp) > SUSPECT_JUMP_C:
                    flag = "suspect"
                    break
        rows.append({
            "station_id": station.id,
            "valid_time": pd.Timestamp(ts),
            "temp_c": temp,
            "obs_time": pd.NaT if obs_ns is None else pd.Timestamp(obs_ns, tz="UTC"),
            "source": source,
            "n_reports": min(n, 127),
            "qc_flag": flag,
            "schema_version": SCHEMA_VERSION,
            "methodology_version": METHODOLOGY_VERSION,
        })
    df = pd.DataFrame.from_records(rows, columns=TRUTH_INSTANT_COLUMNS)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)
    df["temp_c"] = df["temp_c"].astype("float32")
    df["n_reports"] = df["n_reports"].astype("int8")
    return df


def synoptic_times(start, end) -> pd.DatetimeIndex:
    """Every 00/06/12/18 UTC instant inside ``[start, end]`` (dates count as whole UTC days)."""
    lo, hi = _bounds(start, False), _bounds(end, True)
    idx = pd.date_range(lo.normalize(), hi, freq="6h", tz="UTC")
    idx = idx[(idx >= lo) & (idx <= hi)]
    return idx[idx.hour.isin(SYNOPTIC_HOURS)]


def _bounds(value, is_end: bool) -> pd.Timestamp:
    if isinstance(value, date) and not isinstance(value, datetime):
        ts = pd.Timestamp(value, tz="UTC")
        return ts + pd.Timedelta(hours=18) if is_end else ts
    ts = pd.Timestamp(value)
    return ts.tz_convert(UTC) if ts.tzinfo else ts.tz_localize(UTC)


# --------------------------------------------------------------------------- sources

#: reports are fetched this far outside the instant range, so the ±1 h ``suspect`` check has
#: neighbours at the first and last instant too
_PAD = pd.Timedelta(minutes=WINDOW_MIN + 60)


def truth_instant_for_range(
    stations: list[Station] | None = None, start=None, end=None, *, max_workers: int = 3,
) -> pd.DataFrame:
    """``truth_instant`` rows for ``[start, end]`` from the IEM ASOS archive (``source=ASOS_IEM``)."""
    if start is None or end is None:
        raise TypeError("start and end are required")
    stations = list(stations if stations is not None else load_stations())
    times = synoptic_times(start, end)
    if not stations or len(times) == 0:
        return empty_truth_instant()
    lo, hi = times[0] - _PAD, times[-1] + _PAD

    def one(st: Station) -> pd.DataFrame:
        try:
            reports = fetch_asos(st, lo, hi)
        except Exception as exc:  # noqa: BLE001 — one station's outage must not lose the batch
            log.warning("IEM ASOS failed for %s: %s", st.id, type(exc).__name__)
            reports = None
        return instant_from_reports(reports, st, times, source=SOURCE_IEM)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        frames = list(pool.map(one, stations))
    return _concat(frames)


def truth_instant_recent(
    stations: list[Station] | None = None, days: int = 7, *, now: datetime | None = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """``truth_instant`` rows for the last ``days`` from api.weather.gov (``source=NWS_API``).

    Covers the window the IEM archive has not ingested yet.
    """
    end = _bounds(now if now is not None else datetime.now(UTC), False).floor("h")
    start = end - pd.Timedelta(days=max(1, int(days)))
    return _from_nws(stations, synoptic_times(start, end), max_workers=max_workers)


def truth_instant_for_day(stations: list[Station] | None = None, day: date | None = None,
                          **kw) -> pd.DataFrame:
    """The four instants of one UTC day, from api.weather.gov (used by the daily pipeline)."""
    if day is None:
        raise TypeError("day is required")
    return _from_nws(stations, synoptic_times(day, day), **kw)


def _from_nws(stations, times, *, max_workers: int = 4) -> pd.DataFrame:
    """Resolve `times` from api.weather.gov observations, one request per station.

    The API also republishes the 5-minute stream at many sites; those reports are whole °C and land
    on unscheduled minutes, so the :data:`PREFERRED_MINUTES` rule keeps them from displacing the
    routine METAR. Reports carrying a disqualifying quality-control flag are dropped first.
    """
    stations = list(stations if stations is not None else load_stations())
    if not stations or len(times) == 0:
        return empty_truth_instant()
    lo, hi = times[0] - _PAD, times[-1] + _PAD

    def one(st: Station) -> pd.DataFrame:
        obs = _nws_reports(st, lo, hi)
        reports = None
        if obs is not None and len(obs):
            keep = obs[~obs["qc"].fillna("").isin(BAD_QC)]
            reports = keep.rename(columns={"time": "obs_time"})[["obs_time", "temp_c"]]
        return instant_from_reports(reports, st, times, source=SOURCE_NWS)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        frames = list(pool.map(one, stations))
    return _concat(frames)


#: api.weather.gov returns at most a few hundred observations per request and silently truncates to
#: the newest ones, so a week-long window comes back covering only its last two days. Requests are
#: split into chunks no longer than this; a day of 5-minute reports stays inside the cap.
_NWS_CHUNK = pd.Timedelta(days=1)


def _nws_reports(station: Station, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame | None:
    """Observations for ``[lo, hi]`` from api.weather.gov, requested in day-sized chunks."""
    frames = []
    a = lo
    while a <= hi:
        b = min(hi, a + _NWS_CHUNK)
        try:
            frames.append(fetch_hourly_obs(station, a, b))
        except Exception as exc:  # noqa: BLE001 — one chunk's outage must not lose the station
            log.warning("api.weather.gov obs failed for %s %s..%s: %s", station.id, a, b,
                        type(exc).__name__)
        a = b + pd.Timedelta(seconds=1)
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return empty_truth_instant()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["station_id", "valid_time"]).reset_index(drop=True)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per station and year: rows, rows with a value, coverage fraction and the qc-flag counts."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["station_id", "year", "n", "n_value", "coverage",
                                     "no_report", "gap_gt35min", "suspect"])
    d = df.copy()
    d["year"] = pd.to_datetime(d["valid_time"], utc=True).dt.year
    d["_v"] = d["temp_c"].notna()
    g = d.groupby(["station_id", "year"], sort=True)
    out = g.agg(n=("temp_c", "size"), n_value=("_v", "sum")).reset_index()
    for flag in ("no_report", "gap_gt35min", "suspect"):
        counts = d[d["qc_flag"] == flag].groupby(["station_id", "year"]).size()
        out[flag] = out.set_index(["station_id", "year"]).index.map(counts).fillna(0).astype(int)
    out["coverage"] = out["n_value"] / out["n"].where(out["n"] > 0)
    return out


__all__ = [
    "PREFERRED_MINUTES",
    "SOURCE_IEM",
    "SOURCE_NWS",
    "SUSPECT_JUMP_C",
    "SYNOPTIC_HOURS",
    "WINDOW_MIN",
    "coverage",
    "empty_truth_instant",
    "instant_from_reports",
    "synoptic_times",
    "truth_instant_for_day",
    "truth_instant_for_range",
    "truth_instant_recent",
]
