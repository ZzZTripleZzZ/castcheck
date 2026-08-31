"""Derive daily station forecasts and instantaneous errors from extracted station values.

Input is the long ``forecast_values`` table (DESIGN §3.1); outputs are

* ``daily_forecasts`` (DESIGN §3.3) — one row per ``(model_id, init_time, station_id, climo_date,
  method)``, produced by :func:`daily_from_values` / :func:`derive_window`;
* the *instantaneous error* table (DESIGN §10.2) — one row per
  ``(model_id, model_version, init_time, station_id, valid_time, method)``, produced by
  :func:`instant_errors`.  This is the v0.3 headline: the forecast 2 m temperature at one of the four
  common instants against the observation *at that same instant*, which is the only 2 m temperature
  number on this site that is free of any extreme-sampling definition.

Daily extremes (METHODOLOGY §2.3–2.4)
-------------------------------------
* **sampled forecast** ``tmax_sampled_c/tmin_sampled_c`` — max/min of the four common samples
  00/06/12/18 UTC that fall inside the climatological day.  If fewer than four samples are present
  the row is emitted with NaN values, ``n_samples`` < 4 and ``missing_reason="incomplete_samples"``.
* **sampled observation** ``tmax_obs_s_c/tmin_obs_s_c`` (v0.3, new) — max/min of the four *observed*
  values at the very same four instants, from ``truth_instant``.  Verifying the sampled forecast
  extreme against this like-for-like observed extreme is what removes the sampling penalty that the
  external review (``docs/06-external-review-v02.md`` A2) showed was dominating the v0.2 Tmin
  headline.  ``n_obs_samples`` < 4 leaves both NaN, exactly as for the forecast side.
* **native** (diagnostic) — max/min over the model's own time-window extreme fields
  (``mx2t3/mn2t3``, ``mx2t6/mn2t6``, ``tmax6/tmin6``).  METHODOLOGY §2.4: the day is covered by the
  *contiguous run of buckets that overlaps it*, i.e. every bucket lying inside the day plus at most
  one crossing bucket at each end.  The run must be gap-free, must cover the whole day, and may
  overhang it by at most :data:`MAX_OVERHANG_H` hours in total.  v0.3 publishes the *realised*
  overhang of each row as ``native_overhang_h`` so that a reader can see, per station-day, how much
  of the native extreme came from outside the climatological day; :func:`native_overhang_hours`
  gives the same quantity in closed form from the station offset and the bucket length.

  The pre-0.2 rule required the buckets to lie *entirely* inside the day, which only ever happened
  for −6 h (CST) stations: 3 h and 6 h buckets are anchored to 00 UTC, so at −5/−7/−8 h no bucket set
  can tile a day starting at 05/07/08 UTC and every one of those 15 stations got NaN.

Everything is vectorised over the whole table; the only per-run Python work is the (cached)
enumeration of covered climatological days.

Memory
------
:func:`daily_from_values` is a pure function of whatever frame it is handed, but re-deriving the
whole archive every day is O(archive): the full ``forecast_values`` table already costs ~1.9 GB of
resident memory and grows ~3 GB per month, which overruns a 16 GB CI runner within months.
:func:`derive_window` is the incremental entry point — it reads only the shards whose
*initialisation* date falls in the requested window and projects away the five columns the
derivation never reads.  ``daily_forecasts`` is keyed by ``init_time``, so a window re-derivation
rewrites exactly the rows those initialisations produced and leaves the rest of the year alone
(``store.write_daily`` merges per ``(model_id, year)``).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION
from .climo_day import COMMON_SAMPLE_HOURS_UTC, climo_dates_for_run
from .config import DATA_DIR, ModelSpec, Station, load_models, load_stations
from .store import DAILY_COLUMNS

log = logging.getLogger(__name__)

__all__ = [
    "DAILY_V03_COLUMNS",
    "DERIVE_VALUE_COLUMNS",
    "INSTANT_ERROR_COLUMNS",
    "MAX_OVERHANG_H",
    "TRUTH_INSTANT_COLUMNS",
    "daily_columns",
    "daily_from_values",
    "derive_window",
    "empty_daily",
    "empty_instant_errors",
    "extreme_kind",
    "instant_errors",
    "native_overhang_hours",
    "observed_sampled_extremes",
    "read_truth_instant",
]

_KEYS = ["model_id", "init_time", "station_id", "method"]

#: Largest total overhang (hours outside the climatological day) a native-extreme bucket run may
#: have and still be published (METHODOLOGY §2.4).  With 3 h and 6 h buckets and whole-hour station
#: offsets the realised overhang is 0, 3 or 6 h, so this only guards against exotic bucket lengths.
MAX_OVERHANG_H = 6.0

#: v0.3 additions to ``daily_forecasts`` (DESIGN §3.3).  Declared here rather than in
#: :mod:`castcheck.store` so that this module keeps working both before and after the store constant
#: grows them; :func:`daily_columns` is the authoritative output order either way.
DAILY_V03_COLUMNS = ["tmax_obs_s_c", "tmin_obs_s_c", "n_obs_samples", "native_overhang_h"]

#: The only ``forecast_values`` columns the derivation reads.  ``source_url``/``fetched_at`` are the
#: two widest columns in the table and are never touched here.
DERIVE_VALUE_COLUMNS = [
    "model_id", "model_version", "init_time", "valid_time", "station_id", "variable",
    "bucket_h", "method", "value_c", "missing_reason",
]

#: ``truth_instant`` (DESIGN §10.1), owned by :mod:`castcheck.truth` / :mod:`castcheck.store`.
TRUTH_INSTANT_COLUMNS = [
    "station_id", "valid_time", "temp_c", "obs_time", "source", "n_reports", "qc_flag",
    "schema_version", "methodology_version",
]

#: DESIGN §10.2 instantaneous error table.  ``qc_flag`` rides along so that :mod:`castcheck.verify`
#: can count flagged days (METHODOLOGY §6) without re-joining the truth.
INSTANT_ERROR_COLUMNS = [
    "model_id", "model_version", "init_time", "station_id", "valid_time", "method",
    "fcst_c", "obs_c", "err_c", "lead_h", "valid_hour_utc", "lead_day", "climo_date", "qc_flag",
]


def daily_columns() -> list[str]:
    """Output column order of ``daily_forecasts``, v0.3 columns included.

    ``castcheck.store.DAILY_COLUMNS`` is the on-disk contract and is owned by the storage layer; the
    v0.3 columns are appended here (before the two version columns) if that constant has not caught
    up yet, so that this module produces the full v0.3 frame either way.
    """
    cols = list(DAILY_COLUMNS)
    extra = [c for c in DAILY_V03_COLUMNS if c not in cols]
    if not extra:
        return cols
    i = cols.index("schema_version") if "schema_version" in cols else len(cols)
    return [*cols[:i], *extra, *cols[i:]]


def empty_daily() -> pd.DataFrame:
    """An empty frame with exactly the ``daily_forecasts`` columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in daily_columns()})


def empty_instant_errors() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in INSTANT_ERROR_COLUMNS})


def extreme_kind(variable: str) -> str | None:
    """``"max"``/``"min"`` for a native time-window extreme field, else ``None``.

    Accepts the ECMWF (``mx2t3``/``mn2t6``), GFS (``tmax6``/``tmin6``) and GRIB (``TMAX``/``TMIN``)
    spellings so the derivation does not have to know which adapter produced the row.
    """
    v = str(variable).lower()
    if v == "t2":
        return None
    if v.startswith(("mx2t", "tmax")):
        return "max"
    if v.startswith(("mn2t", "tmin")):
        return "min"
    return None


def native_overhang_hours(std_offset_h: int, bucket_h: int) -> float:
    """Hours by which the native bucket run overhangs the climatological day (METHODOLOGY §2.4).

    Buckets are anchored to 00 UTC, so the day ``[-offset, -offset+24)`` UTC is covered by the run
    that starts at the last bucket boundary at or before ``-offset`` and ends at the first boundary
    at or after ``-offset+24``.  The result depends only on the station offset and the bucket length,
    not on the date — 0 h for a −6 h station with 3 h or 6 h buckets, 3 h for −5/−7/−8 h with 3 h
    buckets, 6 h for −5/−7/−8 h with 6 h buckets.  :func:`daily_from_values` publishes the value
    actually realised by each row as ``native_overhang_h``; the two agree whenever the run uses a
    single bucket length.
    """
    if bucket_h <= 0:
        return float("nan")
    r = (-int(std_offset_h)) % int(bucket_h)
    return 0.0 if r == 0 else float(bucket_h)


def _offsets(stations: list[Station]) -> dict[str, int]:
    return {s.id: int(s.std_offset_h) for s in stations}


def _max_h(models: list[ModelSpec]) -> dict[str, int]:
    return {m.model_id: int(m.max_h) for m in models}


def _covered_dates(
    stations: dict[str, Station], max_h: dict[str, int], triples: pd.DataFrame
) -> pd.DataFrame:
    """``(model_id, init_time, station_id)`` → list of covered climatological dates."""
    cache: dict[tuple[str, pd.Timestamp, int], list[date]] = {}
    out: list[list[date]] = []
    for model_id, init_time, station_id in zip(
        triples["model_id"], triples["init_time"], triples["station_id"]
    ):
        st = stations.get(station_id)
        if st is None:
            out.append([])
            continue
        horizon = max_h.get(model_id, 240)
        key = (station_id, pd.Timestamp(init_time), horizon)
        dates = cache.get(key)
        if dates is None:
            dates = climo_dates_for_run(st, pd.Timestamp(init_time).to_pydatetime(), horizon)
            cache[key] = dates
        out.append(dates)
    res = triples.copy()
    res["climo_date"] = out
    return res


def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def _run_versions(v: pd.DataFrame) -> pd.DataFrame:
    """One ``model_version`` per ``(model_id, init_time)``: the modal known value of the run."""
    return (
        v[v["model_version"].astype(str) != "unknown"]
        .groupby(["model_id", "init_time"], observed=True)["model_version"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "unknown")
        .rename("model_version")
        .reset_index()
    )


# --------------------------------------------------------------------------------------------
# truth_instant (DESIGN §10.1) — reader bridge and observed sampled extremes
# --------------------------------------------------------------------------------------------

def read_truth_instant(years: list[int] | None = None) -> pd.DataFrame:
    """Read the ``truth_instant`` table, preferring :func:`castcheck.store.read_truth_instant`.

    The table and its canonical reader are owned by the storage layer (DESIGN §10.1).  Until that
    reader exists this falls back to reading ``data/truth_instant/year=<YYYY>.parquet`` directly with
    the same semantics, so that the derivation and the tests can run against a partially landed
    v0.3.  Returns an empty, correctly typed frame when the table is absent.
    """
    from . import store

    reader = getattr(store, "read_truth_instant", None)
    if reader is not None:
        return reader(years) if years is not None else reader()

    base = DATA_DIR / "truth_instant"
    files = sorted(base.glob("year=*.parquet")) if base.exists() else []
    if years is not None:
        want = {f"year={int(y)}.parquet" for y in years}
        files = [f for f in files if f.name in want]
    if not files:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in TRUTH_INSTANT_COLUMNS})
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["valid_time"] = _to_utc(df["valid_time"])
    return df


def observed_sampled_extremes(
    truth_instant: pd.DataFrame, stations: list[Station] | None = None
) -> pd.DataFrame:
    """Observed extremes over the *same four instants* the forecast is sampled at (§2.3, v0.3).

    Returns ``station_id, climo_date, tmax_obs_s_c, tmin_obs_s_c, n_obs_samples``.  Only days with
    all four observations get a value: a day missing one sample is not the same functional as a day
    with four, and mixing them would reintroduce the definition drift the review objected to.
    """
    cols = ["station_id", "climo_date", "tmax_obs_s_c", "tmin_obs_s_c", "n_obs_samples"]
    if truth_instant is None or len(truth_instant) == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    stations = list(stations) if stations is not None else load_stations()
    offsets = _offsets(stations)
    t = truth_instant.copy()
    t["valid_time"] = _to_utc(t["valid_time"])
    t = t[t["station_id"].isin(offsets)]
    t["temp_c"] = pd.to_numeric(t["temp_c"], errors="coerce")
    t = t[t["temp_c"].notna() & t["valid_time"].dt.hour.isin(COMMON_SAMPLE_HOURS_UTC)]
    if t.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    t = t.drop_duplicates(subset=["station_id", "valid_time"])
    off = pd.to_timedelta(t["station_id"].map(offsets).astype(int), unit="h")
    t["_day"] = (t["valid_time"] + off).dt.floor("D")
    g = (
        t.groupby(["station_id", "_day"], observed=True)["temp_c"]
        .agg(tmax_obs_s_c="max", tmin_obs_s_c="min", n_obs_samples="count")
        .reset_index()
    )
    incomplete = g["n_obs_samples"] < len(COMMON_SAMPLE_HOURS_UTC)
    g.loc[incomplete, ["tmax_obs_s_c", "tmin_obs_s_c"]] = np.nan
    g["climo_date"] = g["_day"].dt.date
    g["n_obs_samples"] = g["n_obs_samples"].clip(upper=4).astype("int8")
    for c in ("tmax_obs_s_c", "tmin_obs_s_c"):
        g[c] = g[c].astype("float32")
    return g[cols].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# instantaneous errors (DESIGN §10.2) — the v0.3 headline
# --------------------------------------------------------------------------------------------

def instant_errors(
    values: pd.DataFrame,
    truth_instant: pd.DataFrame,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
) -> pd.DataFrame:
    """Forecast-minus-observed 2 m temperature at the four common instants.

    One row per ``(model_id, model_version, init_time, station_id, valid_time, method)`` with
    ``fcst_c, obs_c, err_c, lead_h, valid_hour_utc, lead_day, climo_date, qc_flag``.

    Only instants belonging to a climatological day the run covers *completely* are emitted (the
    same rule :func:`daily_from_values` applies, METHODOLOGY §2.5), so the four rows of a
    ``(model, init, station, lead_day, method)`` group are always the four instants of one day and
    the pooled ``t2`` score and the per-hour ``t2_00z…`` scores rest on the same day sample.  Rows
    without a usable observation are dropped: an error needs both sides.
    """
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()
    st_by_id = {s.id: s for s in stations}
    offsets = _offsets(stations)
    max_h = _max_h(models)

    if values is None or len(values) == 0 or truth_instant is None or len(truth_instant) == 0:
        return empty_instant_errors()

    v = values.copy()
    if "bucket_h" not in v:
        v["bucket_h"] = 0
    if "missing_reason" not in v:
        v["missing_reason"] = ""
    v["init_time"] = _to_utc(v["init_time"])
    v["valid_time"] = _to_utc(v["valid_time"])
    v["value_c"] = pd.to_numeric(v["value_c"], errors="coerce")
    v["bucket_h"] = pd.to_numeric(v["bucket_h"], errors="coerce").fillna(0).astype(int)
    v["missing_reason"] = v["missing_reason"].fillna("")
    v = v[
        v["station_id"].isin(st_by_id)
        & (v["variable"] == "t2")
        & (v["bucket_h"] == 0)
        & (v["missing_reason"] == "")
        & v["value_c"].notna()
        & v["valid_time"].dt.hour.isin(COMMON_SAMPLE_HOURS_UTC)
    ]
    if v.empty:
        return empty_instant_errors()
    v = v.drop_duplicates(subset=["model_id", "init_time", "valid_time", "station_id", "method"])

    ti = truth_instant.copy()
    ti["valid_time"] = _to_utc(ti["valid_time"])
    ti["temp_c"] = pd.to_numeric(ti["temp_c"], errors="coerce")
    if "qc_flag" not in ti:
        ti["qc_flag"] = ""
    ti = ti[ti["temp_c"].notna()]
    ti = ti.drop_duplicates(subset=["station_id", "valid_time"])
    ti = ti[["station_id", "valid_time", "temp_c", "qc_flag"]].rename(columns={"temp_c": "obs_c"})

    m = v.merge(ti, on=["station_id", "valid_time"], how="inner")
    if m.empty:
        return empty_instant_errors()

    off = pd.to_timedelta(m["station_id"].map(offsets).astype(int), unit="h")
    m["_day"] = (m["valid_time"] + off).dt.floor("D")

    triples = m[["model_id", "init_time", "station_id"]].drop_duplicates().reset_index(drop=True)
    covered = _covered_dates(st_by_id, max_h, triples).explode("climo_date", ignore_index=True)
    covered = covered[covered["climo_date"].notna()]
    if covered.empty:
        return empty_instant_errors()
    covered["_day"] = pd.to_datetime(covered["climo_date"].astype("str")).dt.tz_localize("UTC")
    m = m.merge(
        covered[["model_id", "init_time", "station_id", "_day"]].drop_duplicates(),
        on=["model_id", "init_time", "station_id", "_day"], how="inner",
    )
    if m.empty:
        return empty_instant_errors()

    ver = _run_versions(v)
    m = m.drop(columns=["model_version"]).merge(ver, on=["model_id", "init_time"], how="left")
    m["model_version"] = m["model_version"].fillna("unknown")

    m["fcst_c"] = m["value_c"].astype("float32")
    m["obs_c"] = m["obs_c"].astype("float32")
    m["err_c"] = (m["fcst_c"] - m["obs_c"]).astype("float32")
    m["lead_h"] = ((m["valid_time"] - m["init_time"]).dt.total_seconds() / 3600.0).astype("int16")
    m["valid_hour_utc"] = m["valid_time"].dt.hour.astype("int8")
    m["lead_day"] = (
        (m["_day"] - m["init_time"].dt.floor("D")).dt.days.astype("int16").astype("int8")
    )
    m["climo_date"] = m["_day"].dt.date
    m["qc_flag"] = m["qc_flag"].fillna("").astype(str)
    out = m[INSTANT_ERROR_COLUMNS].sort_values(
        ["model_id", "init_time", "station_id", "method", "valid_time"]
    )
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# daily_forecasts
# --------------------------------------------------------------------------------------------

def daily_from_values(
    values: pd.DataFrame,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
    truth_instant: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse ``forecast_values`` into the ``daily_forecasts`` table.

    Parameters
    ----------
    values:
        Long table with the DESIGN §3.1 columns.  Rows whose ``missing_reason`` is non-empty (or whose
        ``value_c`` is NaN) still define which ``(model, init, station, method)`` combinations exist,
        but never contribute a value.
    stations, models:
        Registries; default to ``config.load_stations()`` / ``config.load_models()``.
    truth_instant:
        ``truth_instant`` rows (DESIGN §10.1) used to fill the observed sampled extremes
        ``tmax_obs_s_c/tmin_obs_s_c`` and ``n_obs_samples``.  ``None`` leaves those columns empty —
        the frame is still valid, it simply cannot support the ``tmax_s``/``tmin_s`` scores.
    """
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()
    st_by_id = {s.id: s for s in stations}
    offsets = _offsets(stations)
    max_h = _max_h(models)
    out_cols = daily_columns()

    if values is None or len(values) == 0:
        return empty_daily()

    v = values.copy()
    v["init_time"] = _to_utc(v["init_time"])
    v["valid_time"] = _to_utc(v["valid_time"])
    v = v[v["station_id"].isin(st_by_id)]
    if v.empty:
        return empty_daily()
    v["value_c"] = pd.to_numeric(v["value_c"], errors="coerce")
    if "missing_reason" not in v:
        v["missing_reason"] = ""
    v["missing_reason"] = v["missing_reason"].fillna("")
    if "bucket_h" not in v:
        v["bucket_h"] = 0
    v["bucket_h"] = pd.to_numeric(v["bucket_h"], errors="coerce").fillna(0).astype(int)

    offs = v["station_id"].map(offsets).astype(int)
    v["_off"] = pd.to_timedelta(offs, unit="h")

    # ---- the rows we must emit ------------------------------------------------------------
    combos = v[_KEYS].drop_duplicates().reset_index(drop=True)
    triples = combos[["model_id", "init_time", "station_id"]].drop_duplicates().reset_index(drop=True)
    triples = _covered_dates(st_by_id, max_h, triples)
    expected = combos.merge(triples, on=["model_id", "init_time", "station_id"], how="left")
    expected = expected.explode("climo_date", ignore_index=True)
    expected = expected[expected["climo_date"].notna()].reset_index(drop=True)
    if expected.empty:
        return empty_daily()
    expected["_day"] = pd.to_datetime(expected["climo_date"].astype("str")).dt.tz_localize("UTC")

    present = v[(v["missing_reason"] == "") & v["value_c"].notna()]

    # ---- sampled extremes (METHODOLOGY §2.3) ---------------------------------------------
    t2 = present[(present["variable"] == "t2") & (present["bucket_h"] == 0)]
    t2 = t2[t2["valid_time"].dt.hour.isin(COMMON_SAMPLE_HOURS_UTC)]
    if not t2.empty:
        day = (t2["valid_time"] + t2["_off"]).dt.floor("D")
        t2 = t2.assign(_day=day)
        t2 = t2.drop_duplicates(subset=_KEYS + ["_day", "valid_time"])
        samp = (
            t2.groupby(_KEYS + ["_day"], observed=True)["value_c"]
            .agg(tmax_sampled_c="max", tmin_sampled_c="min", n_samples="count")
            .reset_index()
        )
    else:
        samp = pd.DataFrame(columns=_KEYS + ["_day", "tmax_sampled_c", "tmin_sampled_c", "n_samples"])

    # ---- native extremes (METHODOLOGY §2.4, diagnostic) -----------------------------------
    nat = present[present["bucket_h"] > 0].copy()
    if not nat.empty:
        nat["_kind"] = nat["variable"].map(extreme_kind)
        nat = nat[nat["_kind"].notna()]
    if not nat.empty:
        nat["_b_end"] = nat["valid_time"]
        nat["_b_start"] = nat["_b_end"] - pd.to_timedelta(nat["bucket_h"], unit="h")
        # a bucket shorter than 24 h overlaps at most two climatological days: the one containing its
        # start and the one containing its last instant.  Emit the bucket against both.
        cand = []
        for anchor in (nat["_b_start"], nat["_b_end"] - pd.Timedelta(nanoseconds=1)):
            part = nat.copy()
            part["_day"] = (anchor + nat["_off"]).dt.floor("D")
            cand.append(part)
        nat = pd.concat(cand, ignore_index=True)
        nat = nat.drop_duplicates(
            subset=_KEYS + ["_day", "_kind", "bucket_h", "_b_start", "_b_end"]
        )
        nat["_dstart"] = nat["_day"] - nat["_off"]
        nat["_dend"] = nat["_dstart"] + pd.Timedelta(hours=24)

    if not nat.empty:
        gk = _KEYS + ["_day", "_kind"]
        agg = (
            nat.groupby(gk, observed=True)
            .agg(
                _max=("value_c", "max"),
                _min=("value_c", "min"),
                _hours=("bucket_h", "sum"),
                _first=("_b_start", "min"),
                _last=("_b_end", "max"),
                _dstart=("_dstart", "first"),
                _dend=("_dend", "first"),
            )
            .reset_index()
        )
        span_h = (agg["_last"] - agg["_first"]).dt.total_seconds() / 3600.0
        lead_h = (agg["_dstart"] - agg["_first"]).dt.total_seconds() / 3600.0
        trail_h = (agg["_last"] - agg["_dend"]).dt.total_seconds() / 3600.0
        # `span == Σ bucket_h` is true exactly when the buckets form a gap-free, non-overlapping run;
        # it also rejects a day that is offered twice at two bucket lengths (e.g. mx2t3 and mx2t6 for
        # the same hours), while still accepting the IFS day that straddles the 144 h 3 h→6 h change.
        ok = (
            (span_h == agg["_hours"])
            & (lead_h >= 0) & (trail_h >= 0)              # the whole day is covered
            & (lead_h + trail_h <= MAX_OVERHANG_H)
        )
        agg = agg[ok].copy()
        agg["_val"] = np.where(agg["_kind"] == "max", agg["_max"], agg["_min"])
        agg["_overhang"] = (lead_h + trail_h)[ok]
        nat_wide = agg.pivot_table(
            index=_KEYS + ["_day"], columns="_kind", values="_val", aggfunc="first"
        ).reset_index()
        nat_wide = nat_wide.rename(columns={"max": "tmax_native_c", "min": "tmin_native_c"})
        nat_wide.columns.name = None
        over = (
            agg.groupby(_KEYS + ["_day"], observed=True)["_overhang"].max()
            .rename("native_overhang_h").reset_index()
        )
        nat_wide = nat_wide.merge(over, on=_KEYS + ["_day"], how="left")
    else:
        nat_wide = pd.DataFrame(
            columns=_KEYS + ["_day", "tmax_native_c", "tmin_native_c", "native_overhang_h"]
        )
    for c in ("tmax_native_c", "tmin_native_c", "native_overhang_h"):
        if c not in nat_wide.columns:
            nat_wide[c] = np.nan

    # ---- assemble -------------------------------------------------------------------------
    out = expected.merge(samp, on=_KEYS + ["_day"], how="left")
    out = out.merge(
        nat_wide[_KEYS + ["_day", "tmax_native_c", "tmin_native_c", "native_overhang_h"]],
        on=_KEYS + ["_day"], how="left",
    )

    out["n_samples"] = out["n_samples"].fillna(0).clip(upper=4).astype("int8")
    incomplete = out["n_samples"] < 4
    out.loc[incomplete, ["tmax_sampled_c", "tmin_sampled_c"]] = np.nan
    out["missing_reason"] = np.where(incomplete, "incomplete_samples", "")

    # ---- observed sampled extremes (v0.3, DESIGN §10.2) -----------------------------------
    obs_s = observed_sampled_extremes(truth_instant, stations)
    if len(obs_s):
        obs_s = obs_s.copy()
        obs_s["_day"] = pd.to_datetime(obs_s["climo_date"].astype("str")).dt.tz_localize("UTC")
        out = out.merge(
            obs_s[["station_id", "_day", "tmax_obs_s_c", "tmin_obs_s_c", "n_obs_samples"]],
            on=["station_id", "_day"], how="left",
        )
    else:
        out["tmax_obs_s_c"] = np.nan
        out["tmin_obs_s_c"] = np.nan
        out["n_obs_samples"] = 0
    out["n_obs_samples"] = out["n_obs_samples"].fillna(0).clip(upper=4).astype("int8")

    ver = _run_versions(v)
    out = out.merge(ver, on=["model_id", "init_time"], how="left")
    out["model_version"] = out["model_version"].fillna("unknown")

    out["lead_day"] = (
        (out["_day"] - out["init_time"].dt.floor("D")).dt.days.astype("int16").astype("int8")
    )
    out["climo_date"] = out["_day"].dt.date
    for c in ("tmax_sampled_c", "tmin_sampled_c", "tmax_native_c", "tmin_native_c",
              "tmax_obs_s_c", "tmin_obs_s_c", "native_overhang_h"):
        out[c] = out[c].astype("float32")
    out["schema_version"] = SCHEMA_VERSION
    out["methodology_version"] = METHODOLOGY_VERSION
    out = out[out_cols].sort_values(out_cols[:6]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------------
# incremental entry point
# --------------------------------------------------------------------------------------------

def derive_window(
    start_date: date | str,
    end_date: date | str,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
    truth_instant: pd.DataFrame | None = None,
    model_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Derive ``daily_forecasts`` for the initialisations in ``[start_date, end_date]`` (UTC dates).

    The window is over **initialisation** dates, not climatological dates, because that is the key
    ``daily_forecasts`` is sharded and merged on: every row a run produces is rewritten together, so
    a window re-derivation is exactly idempotent and never leaves a half-updated run behind.  A run
    initialised on day *I* covers climatological days *I−1 … I + max_h/24*, so the affected
    climatological days are computed from the runs, not the other way round.

    Only the shards whose init month intersects ``[start_date − 1 day, end_date]`` are opened, and
    only :data:`DERIVE_VALUE_COLUMNS` are read out of them.  On the 2026-08 archive this is ~1/20 of
    the rows and ~2/5 of the columns of a full re-derivation.  The extra leading day absorbs a run
    that landed late and a shard boundary that falls inside the window.
    """
    from .store import read_forecast_values

    start = pd.Timestamp(start_date).date() if not isinstance(start_date, date) else start_date
    end = pd.Timestamp(end_date).date() if not isinstance(end_date, date) else end_date
    read_start = start - timedelta(days=1)
    values = read_forecast_values(
        model_ids=model_ids,
        start=read_start.isoformat(),
        end=end.isoformat(),
        columns=DERIVE_VALUE_COLUMNS,
    )
    log.info(
        "derive_window %s..%s: %d forecast_values rows in %d columns",
        read_start, end, len(values), len(values.columns),
    )
    if len(values) == 0:
        return empty_daily()
    if truth_instant is None:
        years = sorted({int(y) for y in pd.to_datetime(values["init_time"], utc=True).dt.year})
        years = sorted(set(years) | {y + 1 for y in years})
        try:
            truth_instant = read_truth_instant(years)
        except Exception as exc:  # pragma: no cover - defensive: a missing/unreadable table
            log.warning("truth_instant unavailable (%s); observed sampled extremes left empty", exc)
            truth_instant = None
    return daily_from_values(values, stations, models, truth_instant=truth_instant)
