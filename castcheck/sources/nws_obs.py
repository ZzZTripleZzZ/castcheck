"""Hourly station observations from api.weather.gov — fallback truth and QC only (METHODOLOGY §3/§6).

``maxTemperatureLast24Hours`` is always null in this API, so daily extremes have to be derived from
the reported instantaneous temperatures. That under-samples the diurnal cycle: the derived maximum
runs roughly 1 °F below the CLI value. These values are therefore never a first-choice truth; they
fill gaps (flagged) and drive the ``obs_diff_gt2f`` quality flag.

Many ASOS sites publish both 5-minute values rounded to a whole °C *and* the routine METAR reading
at 0.1 °C from the ``Tddd`` group. Mixing them adds up to 0.5 °C of rounding noise, so
:func:`daily_extremes_from_obs` keeps the high-resolution reports when a day has any.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from ..climo_day import day_bounds_utc
from ..config import Station
from .nws_cli import NWS_API, _get

OBS_COLUMNS = ["time", "temp_c", "qc"]

#: quality-control flags that disqualify an observation (see api.weather.gov ontology):
#: ``X`` failed validity, ``B`` subjective bad, ``Q`` questioned, ``Z`` preliminary/no QC.
BAD_QC = frozenset({"X", "B", "Q", "Z"})


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def fetch_hourly_obs(station: Station, start: datetime, end: datetime) -> pd.DataFrame:
    """Observations for ``[start, end]`` as ``time`` (UTC), ``temp_c``, ``qc``; empty frame on error."""
    url = f"{NWS_API}/stations/{station.id}/observations"
    params = {"start": _iso(start), "end": _iso(end)}
    try:
        r = _get(url, params=params, timeout=90)
    except RuntimeError:
        return _empty()
    if r.status_code != 200:
        return _empty()
    try:
        feats = r.json().get("features") or []
    except ValueError:
        return _empty()

    recs = []
    for f in feats:
        p = f.get("properties") or {}
        t = (p.get("temperature") or {})
        recs.append(
            {
                "time": pd.Timestamp(p["timestamp"]).tz_convert("UTC"),
                "temp_c": t.get("value"),
                "qc": t.get("qualityControl") or "",
            }
        )
    if not recs:
        return _empty()
    df = pd.DataFrame.from_records(recs, columns=OBS_COLUMNS)
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _iso(t) -> str:
    ts = pd.Timestamp(t)
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty() -> pd.DataFrame:
    df = pd.DataFrame(columns=OBS_COLUMNS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_obs_day(station: Station, climo_date: date) -> pd.DataFrame:
    """Observations covering the station's climatological day (plus a small margin)."""
    start, end = day_bounds_utc(station, climo_date)
    return fetch_hourly_obs(station, start, end)


def daily_extremes_from_obs(
    obs: pd.DataFrame, station: Station, climo_date: date
) -> tuple[float | None, float | None]:
    """Derived (tmax_f, tmin_f) in **degrees Fahrenheit** for the station's climatological day.

    Observations outside the LST day and those with a disqualifying ``qc`` flag are dropped. When
    the day contains any 0.1 °C METAR readings, only reports at those routine minutes are used so
    that whole-°C 5-minute values cannot distort the extremes. Returns ``(None, None)`` when no
    usable observation remains.
    """
    if obs is None or obs.empty:
        return (None, None)
    start, end = day_bounds_utc(station, climo_date)
    df = obs.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[(df["time"] >= pd.Timestamp(start)) & (df["time"] < pd.Timestamp(end))]
    df = df[~df["qc"].fillna("").isin(BAD_QC)]
    df = df.dropna(subset=["temp_c"])
    if df.empty:
        return (None, None)

    hi_res = df[(df["temp_c"] * 10).round() % 10 != 0]
    if not hi_res.empty:
        minutes = set(hi_res["time"].dt.minute.unique())
        df = df[df["time"].dt.minute.isin(minutes)]
    if df.empty:
        return (None, None)
    return (float(c_to_f(df["temp_c"].max())), float(c_to_f(df["temp_c"].min())))


def obs_extremes_for_day(station: Station, climo_date: date) -> tuple[float | None, float | None]:
    """Convenience wrapper: fetch the day's observations and reduce them to (tmax_f, tmin_f)."""
    return daily_extremes_from_obs(fetch_obs_day(station, climo_date), station, climo_date)


__all__ = [
    "BAD_QC",
    "OBS_COLUMNS",
    "c_to_f",
    "daily_extremes_from_obs",
    "fetch_hourly_obs",
    "fetch_obs_day",
    "obs_extremes_for_day",
]
