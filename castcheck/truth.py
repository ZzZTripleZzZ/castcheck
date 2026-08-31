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
from datetime import date, datetime, timedelta
from functools import partial

import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION
from .climo_day import day_bounds_utc
from .config import Station, load_stations
from .sources.nws_cf6 import fetch_cf6
from .sources.nws_cli import cli_history_by_day, fetch_cli_day
from .sources.nws_obs import c_to_f, daily_extremes_from_obs, fetch_obs_day
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


# --------------------------------------------------------------- plausibility QC (METHODOLOGY §3.3)

#: A day's true maximum can never be *below* the maximum of the four sampled observations, and its
#: true minimum can never be *above* the sampled minimum — the samples are part of the same trace.
#: This tolerance covers what legitimately blurs that identity: whole-°F rounding on both sides and
#: the up-to-35-minute offset between a synoptic instant and the METAR that represents it.
PLAUSIBILITY_TOLERANCE_F = 1.5

#: On the other side the inequality is *expected* to be loose — the true extreme normally lies
#: between two samples (a dawn minimum falls between 06 and 12 UTC), so the CLI value being well
#: beyond the sampled extreme is ordinary, not suspicious. This is how far beyond it may be before
#: the day is treated as a mis-decode rather than as an unsampled excursion. See the note in
#: METHODOLOGY §3.3 on how the value was chosen.
PLAUSIBILITY_EXCURSION_F = 10.0

#: The widest excursion that turned out to be real weather anywhere in the 2024-2026 archive is
#: 25 °F (KOKC 2024-02-27, a February frontal passage: the minimum fell 25 °F below the lowest of
#: the day's four samples, and no correction was ever issued because nothing was wrong). An
#: uncorroborated excursion *beyond* that bound is outside everything three years of observations
#: support, so it is dropped rather than published: keeping a value that far outside the observed
#: envelope costs more, as truth, than losing one station-day costs as coverage. Inside the bound
#: the value is kept untouched — see :func:`plausibility_qc`.
PLAUSIBILITY_REVIEW_F = 25.0

#: qc_flag tokens written by :func:`plausibility_qc`
QC_IMPLAUSIBLE = "cli_implausible"
QC_REVISED_USED = "cli_implausible_revised_used"
QC_CF6_USED = "cli_implausible_cf6_used"
QC_DROPPED = "cli_implausible_dropped"


def _violation(value, obs_f: float, *, is_max: bool) -> str:
    """How a reported daily extreme conflicts with the day's sampled observations.

    ``"impossible"``  the value sits on the wrong side of the sampled extreme by more than
                      :data:`PLAUSIBILITY_TOLERANCE_F`. The four samples are part of the same
                      trace, so a daily maximum *below* the sampled maximum cannot happen.
    ``"excursion"``   the value is further beyond the sampled extreme than
                      :data:`PLAUSIBILITY_EXCURSION_F`. On its own this is not evidence of
                      anything — see :func:`plausibility_qc`.
    ``""``            consistent.
    """
    v = _nn(value)
    if v is None:
        return ""  # nothing to check; missing values are handled by the existing flags
    beyond = (v - obs_f) if is_max else (obs_f - v)   # + when past the sampled extreme
    if beyond < -PLAUSIBILITY_TOLERANCE_F:
        return "impossible"
    return "excursion" if beyond > PLAUSIBILITY_EXCURSION_F else ""


def sampled_extremes_f(truth_instant: pd.DataFrame, stations: list[Station] | None = None) -> dict:
    """``(station_id, climo_date) -> (obs_max_f, obs_min_f)`` from ``truth_instant``.

    Only station-days whose **four** instants are all present and none flagged ``suspect`` are
    returned: a check against three samples would raise a false alarm every time the missing one was
    the extreme, and a check against a sensor that has already been flagged as doubtful would
    condemn the CLI report on the strength of the worse measurement.

    An instant belongs to the climatological day whose local-standard-time midnight precedes it,
    which is ``valid_time + std_offset_h`` — the same mapping :func:`castcheck.climo_day.
    common_sample_times` produces, computed here in one vectorised pass instead of per station-day.
    """
    if truth_instant is None or len(truth_instant) == 0:
        return {}
    offsets = {s.id: int(s.std_offset_h) for s in (stations or load_stations())}
    df = truth_instant[["station_id", "valid_time", "temp_c", "qc_flag"]].copy()
    df = df[df["station_id"].isin(offsets)]
    if df.empty:
        return {}
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    shift = df["station_id"].map(offsets).astype("int64")
    df["climo_date"] = (df["valid_time"] + pd.to_timedelta(shift, unit="h")).dt.date
    df["_bad"] = df["temp_c"].isna() | (df["qc_flag"].fillna("") == "suspect")

    g = df.groupby(["station_id", "climo_date"], sort=False)
    agg = g.agg(n=("temp_c", "size"), bad=("_bad", "sum"),
                hi=("temp_c", "max"), lo=("temp_c", "min"))
    ok = agg[(agg["n"] == 4) & (agg["bad"] == 0)]
    return {key: (c_to_f(hi), c_to_f(lo))
            for key, hi, lo in zip(ok.index, ok["hi"], ok["lo"])}


def plausibility_qc(
    truth_daily: pd.DataFrame, truth_instant: pd.DataFrame, stations: list[Station] | None = None,
) -> pd.DataFrame:
    """Check every published CLI extreme against the day's own observations; repair or drop it.

    **Why this exists.** The first-final policy (§3.2) is what makes the truth reproducible: the
    value is fixed by the first report issued after local midnight and a later correction never
    changes a published score. Its cost, measured on this archive, is that a *garbled* first report
    is scored as truth — KLAX 2025-02-16 was issued with ``MINIMUM 11R`` and corrected to 49 six
    hours later, so the published truth was 38 °F below what every observation that night showed.
    That is not a first-final decision any more, it is a decode error surviving into the scores.

    **What it changes.** Nothing about first-final: the *first* report is still the one consulted,
    and a correction that merely disagrees is still ignored. Only a value that contradicts the
    station's own instantaneous observations is acted on, and then in a fixed order — the correction
    if it passes the same check, else the CF6 monthly table if it passes, else the value is dropped
    (``NaN``) so the day leaves the scores instead of poisoning them. Every outcome is recorded in
    ``qc_flag``, and days with no four-sample coverage are left exactly as they were.

    The frame is returned modified; ``truth_daily`` is not mutated in place. Re-running is a no-op,
    because a repaired value passes the check that flagged it.
    """
    if truth_daily is None or truth_daily.empty:
        return truth_daily
    out = truth_daily.copy()
    obs = sampled_extremes_f(truth_instant, stations)
    if not obs:
        return out
    out["climo_date"] = [d.date() if isinstance(d, datetime) else d
                         for d in pd.to_datetime(out["climo_date"])]

    cf6 = {(r.station_id, r.climo_date): (_nn(r.tmax_f), _nn(r.tmin_f))
           for r in out[out["source"] == "CF6"].itertuples()}

    counts = {QC_REVISED_USED: 0, QC_CF6_USED: 0, QC_DROPPED: 0}
    for i in out.index[out["source"] == "CLI"]:
        key = (out.at[i, "station_id"], out.at[i, "climo_date"])
        bounds = obs.get(key)
        if bounds is None:
            continue
        obs_max_f, obs_min_f = bounds
        flags = [out.at[i, "qc_flag"]]
        for col, rev_col, c_col, is_max, bound in (
            ("tmax_f", "revised_tmax_f", "tmax_c", True, obs_max_f),
            ("tmin_f", "revised_tmin_f", "tmin_c", False, obs_min_f),
        ):
            how = _violation(out.at[i, col], bound, is_max=is_max)
            if not how:
                continue
            alternatives = [(_nn(out.at[i, rev_col]), QC_REVISED_USED),
                            (_cf6_value(cf6.get(key), col), QC_CF6_USED)]
            fix, token = next(
                ((v, tok) for v, tok in alternatives
                 if v is not None and not _violation(v, bound, is_max=is_max)),
                (None, QC_DROPPED),
            )
            beyond = abs(_nn(out.at[i, col]) - bound)
            # round() because both sides live on the whole-°F lattice the reports are published on:
            # a 25 °F excursion arrives here as 24.999999 after the °C round trip, and must compare
            # as the 25 it is rather than being dropped by a floating-point artefact.
            if how == "excursion" and token == QC_DROPPED and round(beyond) <= PLAUSIBILITY_REVIEW_F:
                # An unsampled excursion inside the observed envelope, with nothing to corroborate
                # an error, is just weather: the true extreme legitimately falls between two
                # samples. Dropping it would remove precisely the hardest days from the scores.
                continue
            if how == "excursion" and token == QC_DROPPED:
                LOG.warning("%s %s: %s=%s is %.0f °F beyond the sampled extreme — past the %.0f °F "
                            "envelope of every real excursion in the archive, and neither a "
                            "correction nor CF6 supports it; dropped",
                            key[0], key[1], col, out.at[i, col], beyond, PLAUSIBILITY_REVIEW_F)
            else:
                LOG.warning("%s %s: %s=%s %s against sampled %.1f °F -> %s",
                            key[0], key[1], col, out.at[i, col], how, bound, fix)
            out.at[i, col] = fix
            out.at[i, c_col] = f_to_c(fix)
            flags += [QC_IMPLAUSIBLE, token]
            counts[token] += 1
        joined = _join_flags(*flags)
        out.at[i, "qc_flag"] = joined
        if QC_IMPLAUSIBLE in joined:
            # This row's published value is whatever this check decided it should be, so the row
            # describes itself by the methodology that decided it. Keyed off the *flag* rather than
            # off "did this pass change something", because a repaired value passes the check on
            # every later pass and would otherwise keep the version of the rule it no longer obeys.
            out.at[i, "methodology_version"] = METHODOLOGY_VERSION
    if any(counts.values()):
        LOG.info("plausibility QC: %d revised, %d CF6, %d dropped",
                 counts[QC_REVISED_USED], counts[QC_CF6_USED], counts[QC_DROPPED])
    for col in _INT_COLS:
        out[col] = pd.array([_nn(v) for v in out[col]], dtype="Int16")
    return out


def _cf6_value(pair, col: str):
    if pair is None:
        return None
    return pair[0] if col == "tmax_f" else pair[1]


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
    return _with_plausibility_qc(_concat(frames), stations)


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
    return _with_plausibility_qc(_concat(frames), stations)


def _with_plausibility_qc(rows: pd.DataFrame, stations: list[Station]) -> pd.DataFrame:
    """Run :func:`plausibility_qc` against whatever ``truth_instant`` already holds.

    Best-effort by design: on a normal daily run the instants for the day being fetched are not
    stored yet (``truth`` runs before ``truth-instant``), so this catches nothing and the standalone
    ``castcheck truth-qc`` pass, which runs after both, is what actually repairs the day. It is
    still worth doing here — a backfill re-run over old dates has the instants available and fixes
    them on the spot — and it must never be able to fail the fetch, which is why the store read is
    guarded.
    """
    if rows is None or rows.empty:
        return rows
    try:
        from .store import read_truth_instant

        years = sorted({d.year for d in pd.to_datetime(rows["climo_date"]).dt.date})
        return plausibility_qc(rows, read_truth_instant(years=years), stations)
    except Exception as exc:  # noqa: BLE001 — QC is a safeguard, never a dependency of the fetch
        LOG.warning("plausibility QC skipped: %s: %s", type(exc).__name__, exc)
        return rows


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
