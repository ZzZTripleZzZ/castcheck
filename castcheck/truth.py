"""Assemble the ``truth_daily`` table from CLI / CF6 / hourly observations (METHODOLOGY §3, §6).

Source precedence for a station-day:

1. **CLI** — the ``YESTERDAY`` block of the *first* Daily Climate Report issued after local
   midnight ("first-final"). Later corrected issuances are recorded in ``revised``/``revised_*``
   and never replace the published value.
2. **CF6** — the Preliminary Monthly Climate Data table, used when no CLI exists for the day.
3. **OBS** — extremes derived from hourly observations, always flagged (``obs_fallback``) because
   hourly sampling misses the true peak by roughly 1 °F.

Hourly observations are additionally used as a cross-check on every day: a disagreement of more
than 2 °F with the chosen source raises ``obs_diff_gt2f``. Flagged days stay in the scores
(METHODOLOGY §6); the flag is published alongside them.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import partial

import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION
from .climo_day import day_bounds_utc
from .config import Station, load_stations
from .sources.nws_cf6 import fetch_cf6
from .sources.nws_cli import cli_history_by_day, fetch_cli_day
from .sources.nws_obs import daily_extremes_from_obs, fetch_obs_day
from .store import TRUTH_COLUMNS

LOG = logging.getLogger(__name__)

#: order in which sources are trusted when more than one row exists for a station-day
SOURCE_PRIORITY = ("CLI", "CF6", "OBS")

#: |CLI − hourly-derived| above this many °F raises ``obs_diff_gt2f`` (METHODOLOGY §6)
OBS_QC_THRESHOLD_F = 2.0

_INT_COLS = ("tmax_f", "tmin_f", "revised_tmax_f", "revised_tmin_f")


def f_to_c(f: float | None) -> float | None:
    """Whole-degree Fahrenheit as reported → °C."""
    return None if f is None else (float(f) - 32.0) * 5.0 / 9.0


def _row(
    *, station: Station, climo_date: date, source: str, tmax_f, tmin_f, issuance_time,
    is_final: bool, revised: bool = False, revised_tmax_f=None, revised_tmin_f=None,
    qc_flag: str = "", product_id: str = "",
) -> dict:
    return {
        "station_id": station.id,
        "climo_date": climo_date,
        "source": source,
        "tmax_f": tmax_f,
        "tmin_f": tmin_f,
        "tmax_c": f_to_c(tmax_f),
        "tmin_c": f_to_c(tmin_f),
        "issuance_time": issuance_time,
        "is_final": bool(is_final),
        "revised": bool(revised),
        "revised_tmax_f": revised_tmax_f,
        "revised_tmin_f": revised_tmin_f,
        "qc_flag": qc_flag,
        "product_id": product_id or "",
        "schema_version": SCHEMA_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
    }


def _obs_flag(tmax_f, tmin_f, obs: tuple[float | None, float | None] | None) -> str:
    """``obs_diff_gt2f`` when the reported extremes disagree with the hourly-derived ones."""
    if not obs:
        return ""
    o_max, o_min = obs
    for reported, derived in ((tmax_f, o_max), (tmin_f, o_min)):
        if reported is None or derived is None:
            continue
        if abs(float(reported) - float(derived)) > OBS_QC_THRESHOLD_F:
            return "obs_diff_gt2f"
    return ""


def _join_flags(*flags: str) -> str:
    seen: list[str] = []
    for f in flags:
        for part in (f or "").split(";"):
            if part and part not in seen:
                seen.append(part)
    return ";".join(seen)


def _cf6_day(cf6, climo_date: date) -> dict | None:
    """Normalise the ``cf6`` argument (DataFrame / row / dict) to one day's dict, or ``None``."""
    if cf6 is None:
        return None
    if isinstance(cf6, pd.DataFrame):
        if cf6.empty:
            return None
        hit = cf6[pd.Series([d == climo_date for d in cf6["climo_date"]], index=cf6.index)]
        if hit.empty:
            return None
        cf6 = hit.iloc[-1]
    if isinstance(cf6, pd.Series):
        cf6 = cf6.to_dict()
    return dict(cf6)


def _nn(v):
    """pandas NA / NaN → ``None``, otherwise an ``int``."""
    if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def build_truth_rows(
    station: Station,
    climo_date: date,
    cli: dict | None = None,
    cf6=None,
    obs: tuple[float | None, float | None] | None = None,
) -> pd.DataFrame:
    """One ``truth_daily`` row per available source for a single station-day.

    ``cli`` is a :func:`castcheck.sources.nws_cli.fetch_cli_day` result (may carry
    ``later_versions``), ``cf6`` a :func:`castcheck.sources.nws_cf6.fetch_cf6` frame/row for the
    month, and ``obs`` the ``(tmax_f, tmin_f)`` pair derived from hourly observations. Every
    argument is optional; an empty frame with the right columns is returned when all are missing.
    """
    _, day_end = day_bounds_utc(station, climo_date)
    rows: list[dict] = []

    if cli:
        tmax_f, tmin_f = _nn(cli.get("tmax_f")), _nn(cli.get("tmin_f"))
        later = [v for v in (cli.get("later_versions") or [])
                 if _nn(v.get("tmax_f")) != tmax_f or _nn(v.get("tmin_f")) != tmin_f]
        flags = []
        if tmax_f is None or tmin_f is None:
            flags.append("cli_missing_value")
        if cli.get("block") != "YESTERDAY":
            flags.append("cli_not_final")
        flags.append(_obs_flag(tmax_f, tmin_f, obs))
        rows.append(
            _row(
                station=station, climo_date=climo_date, source="CLI", tmax_f=tmax_f, tmin_f=tmin_f,
                issuance_time=cli.get("issuance_time") or day_end,
                is_final=bool(cli.get("is_final", cli.get("block") == "YESTERDAY")),
                revised=bool(later) or bool(cli.get("is_corrected")),
                revised_tmax_f=_nn(later[-1].get("tmax_f")) if later else None,
                revised_tmin_f=_nn(later[-1].get("tmin_f")) if later else None,
                qc_flag=_join_flags(*flags),
                product_id=cli.get("product_id") or "",
            )
        )

    day = _cf6_day(cf6, climo_date)
    if day is not None:
        tmax_f, tmin_f = _nn(day.get("tmax_f")), _nn(day.get("tmin_f"))
        flags = ["cf6_missing_value"] if (tmax_f is None or tmin_f is None) else []
        flags.append(_obs_flag(tmax_f, tmin_f, obs))
        issued = day.get("issuance_time")
        rows.append(
            _row(
                station=station, climo_date=climo_date, source="CF6", tmax_f=tmax_f, tmin_f=tmin_f,
                issuance_time=day_end if issued is None or issued is pd.NaT else issued,
                is_final=True, qc_flag=_join_flags(*flags), product_id=str(day.get("product_id") or ""),
            )
        )

    if obs and (obs[0] is not None or obs[1] is not None):
        rows.append(
            _row(
                station=station, climo_date=climo_date, source="OBS",
                tmax_f=None if obs[0] is None else round(obs[0]),
                tmin_f=None if obs[1] is None else round(obs[1]),
                issuance_time=day_end, is_final=False, qc_flag="obs_fallback",
                product_id=f"{station.id}/observations",
            )
        )

    return _frame(rows)


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(rows, columns=TRUTH_COLUMNS)
    for col in _INT_COLS:
        df[col] = pd.array([_nn(v) for v in df[col]], dtype="Int16")
    df["issuance_time"] = pd.to_datetime(df["issuance_time"], utc=True)
    df["is_final"] = df["is_final"].fillna(False).astype(bool)
    df["revised"] = df["revised"].fillna(False).astype(bool)
    for col in ("station_id", "source", "qc_flag", "product_id", "schema_version", "methodology_version"):
        df[col] = df[col].fillna("").astype(str)
    return df


def assemble_truth(cli_rows: pd.DataFrame, cf6_rows: pd.DataFrame, obs_rows: pd.DataFrame) -> pd.DataFrame:
    """DESIGN §4 entry point: concatenate per-source frames into one ``truth_daily`` table.

    Duplicate ``(station_id, climo_date, source)`` keys are reduced to the earliest issuance, which
    is the first-final policy applied in memory; :func:`castcheck.store.upsert_truth` applies the
    same rule against what is already on disk.
    """
    frames = [f for f in (cli_rows, cf6_rows, obs_rows) if f is not None and len(f)]
    if not frames:
        return _frame([])
    df = pd.concat(frames, ignore_index=True)[TRUTH_COLUMNS]
    df = df.sort_values(["station_id", "climo_date", "source", "issuance_time"])
    return df.drop_duplicates(subset=["station_id", "climo_date", "source"], keep="first").reset_index(drop=True)


def best_truth(truth: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``truth_daily`` to one row per station-day using :data:`SOURCE_PRIORITY`."""
    if truth.empty:
        return truth
    df = truth.copy()
    order = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    df["_p"] = df["source"].map(lambda s: order.get(s, len(order)))
    df = df.sort_values(["station_id", "climo_date", "_p", "issuance_time"])
    return df.drop_duplicates(subset=["station_id", "climo_date"], keep="first").drop(columns="_p").reset_index(drop=True)


# --------------------------------------------------------------------------- online


def _safe(what: str, station: Station, fn, default):
    """Run one upstream call; a failure degrades that source instead of losing the whole batch."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a partial fetch must never abort the run
        LOG.warning("%s failed for %s: %s", what, station.id, exc)
        return default


def _truth_one_day(station: Station, climo_date: date, use_obs: bool = True) -> pd.DataFrame:
    cli = _safe("CLI", station, lambda: fetch_cli_day(station, climo_date), None)
    obs = None
    if use_obs:
        obs = _safe(
            "OBS", station,
            lambda: daily_extremes_from_obs(fetch_obs_day(station, climo_date), station, climo_date),
            None,
        )
        if obs == (None, None):
            obs = None
    cf6 = None
    if cli is None or cli.get("tmax_f") is None or cli.get("tmin_f") is None:
        cf6 = _safe("CF6", station, lambda: fetch_cf6(station, climo_date.year, climo_date.month), None)
    return build_truth_rows(station, climo_date, cli=cli, cf6=cf6, obs=obs)


def truth_for_date(
    stations: list[Station] | None = None, climo_date: date | None = None, *,
    use_obs: bool = True, max_workers: int = 6,
) -> pd.DataFrame:
    """Fetch today's truth for every station: CLI first-final, CF6 when CLI is missing, OBS as a
    flagged fallback and as the QC cross-check.

    All rows for all available sources are returned (not just the winning one) so that the stored
    table keeps the evidence; use :func:`best_truth` to reduce it.
    """
    if climo_date is None:
        raise TypeError("climo_date is required")
    stations = list(stations or load_stations())
    if not stations:
        return _frame([])
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(pool.map(lambda s: _truth_one_day(s, climo_date, use_obs), stations))
    return _concat(frames)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return _frame([])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["station_id", "climo_date", "source"]).reset_index(drop=True)


# --------------------------------------------------------------------------- backfill


def _truth_backfill_station(station: Station, start: date, end: date) -> pd.DataFrame:
    cli_by_day = _safe("CLI archive", station, lambda: cli_history_by_day(station, start, end), {}) or {}

    missing = [d for d in _daterange(start, end) if d not in cli_by_day
               or cli_by_day[d].get("tmax_f") is None or cli_by_day[d].get("tmin_f") is None]
    cf6_by_month: dict[tuple[int, int], pd.DataFrame] = {}
    for d in missing:
        key = (d.year, d.month)
        if key not in cf6_by_month:
            cf6_by_month[key] = _safe("CF6", station, partial(fetch_cf6, station, key[0], key[1]), None)

    frames = []
    for d in _daterange(start, end):
        cli = cli_by_day.get(d)
        cf6 = cf6_by_month.get((d.year, d.month)) if d in missing else None
        if cli is None and (cf6 is None or cf6.empty):
            continue
        frames.append(build_truth_rows(station, d, cli=cli, cf6=cf6, obs=None))
    return _concat(frames)


def truth_backfill(
    stations: list[Station] | None = None, start: date | None = None, end: date | None = None, *,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Historic truth for ``[start, end]`` from the IEM AFOS CLI archive, with CF6 filling gaps.

    Hourly observations are not fetched (one request per station-day is prohibitive over long
    ranges), so backfilled rows carry no ``obs_diff_gt2f`` flag.
    """
    if start is None or end is None:
        raise TypeError("start and end are required")
    stations = list(stations or load_stations())
    if not stations:
        return _frame([])
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(pool.map(lambda s: _truth_backfill_station(s, start, end), stations))
    return _concat(frames)


def _daterange(start: date, end: date) -> list[date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def missing_days(truth: pd.DataFrame, stations: list[Station], start: date, end: date) -> pd.DataFrame:
    """Station-days in ``[start, end]`` with no usable truth value; ``source`` is the best found."""
    best = best_truth(truth)
    have = {
        (r.station_id, r.climo_date): r.source
        for r in best.itertuples()
        if not pd.isna(r.tmax_f) and not pd.isna(r.tmin_f)
    }
    gaps = [
        {"station_id": s.id, "climo_date": d}
        for s in stations for d in _daterange(start, end) if (s.id, d) not in have
    ]
    return pd.DataFrame(gaps, columns=["station_id", "climo_date"])
