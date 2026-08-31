"""Derive daily station forecasts from extracted station values (METHODOLOGY §2.3–2.5).

Input is the long ``forecast_values`` table (DESIGN §3.1); output is ``daily_forecasts`` (DESIGN §3.3):
one row per ``(model_id, init_time, station_id, climo_date, method)``.

Two daily extremes are produced per row:

* **sampled** (headline) — max/min of the four common samples 00/06/12/18 UTC that fall inside the
  climatological day (METHODOLOGY §2.3).  If fewer than four samples are present the row is emitted
  with NaN values, ``n_samples`` < 4 and ``missing_reason="incomplete_samples"``.
* **native** (diagnostic only) — max/min over the model's own time-window extreme fields
  (``mx2t3/mn2t3``, ``mx2t6/mn2t6``, ``tmax6/tmin6``).  METHODOLOGY §2.4 (v0.2): the day is covered
  by the *contiguous run of buckets that overlaps it*, i.e. every bucket lying inside the day plus at
  most one crossing bucket at each end.  The run must be gap-free, must cover the whole day, and may
  overhang it by at most :data:`MAX_OVERHANG_H` hours in total; the overhang is a deterministic
  function of the station's standard offset and the bucket length (see
  :func:`native_overhang_hours`).  Models without a native field always get NaN.

  The pre-0.2 rule required the buckets to lie *entirely* inside the day, which only ever happened
  for −6 h (CST) stations: 3 h and 6 h buckets are anchored to 00 UTC, so at −5/−7/−8 h no bucket set
  can tile a day starting at 05/07/08 UTC and every one of those 15 stations got NaN.

Everything is vectorised over the whole table; the only per-run Python work is the (cached)
enumeration of covered climatological days.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION
from .climo_day import COMMON_SAMPLE_HOURS_UTC, climo_dates_for_run
from .config import ModelSpec, Station, load_models, load_stations
from .store import DAILY_COLUMNS

__all__ = [
    "MAX_OVERHANG_H",
    "daily_from_values",
    "empty_daily",
    "extreme_kind",
    "native_overhang_hours",
]

_KEYS = ["model_id", "init_time", "station_id", "method"]

#: Largest total overhang (hours outside the climatological day) a native-extreme bucket run may
#: have and still be published (METHODOLOGY §2.4).  With 3 h and 6 h buckets and whole-hour station
#: offsets the realised overhang is 0, 3 or 6 h, so this only guards against exotic bucket lengths.
MAX_OVERHANG_H = 6.0


def empty_daily() -> pd.DataFrame:
    """An empty frame with exactly the ``daily_forecasts`` columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in DAILY_COLUMNS})


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
    buckets, 6 h for −5/−7/−8 h with 6 h buckets.
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
    s = pd.to_datetime(s, utc=True)
    return s


def daily_from_values(
    values: pd.DataFrame,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
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
    """
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()
    st_by_id = {s.id: s for s in stations}
    offsets = _offsets(stations)
    max_h = _max_h(models)

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
        agg = agg[ok]
        agg["_val"] = np.where(agg["_kind"] == "max", agg["_max"], agg["_min"])
        nat_wide = agg.pivot_table(
            index=_KEYS + ["_day"], columns="_kind", values="_val", aggfunc="first"
        ).reset_index()
        nat_wide = nat_wide.rename(columns={"max": "tmax_native_c", "min": "tmin_native_c"})
        nat_wide.columns.name = None
    else:
        nat_wide = pd.DataFrame(columns=_KEYS + ["_day", "tmax_native_c", "tmin_native_c"])
    for c in ("tmax_native_c", "tmin_native_c"):
        if c not in nat_wide.columns:
            nat_wide[c] = np.nan

    # ---- assemble -------------------------------------------------------------------------
    out = expected.merge(samp, on=_KEYS + ["_day"], how="left")
    out = out.merge(nat_wide[_KEYS + ["_day", "tmax_native_c", "tmin_native_c"]], on=_KEYS + ["_day"], how="left")

    out["n_samples"] = out["n_samples"].fillna(0).clip(upper=4).astype("int8")
    incomplete = out["n_samples"] < 4
    out.loc[incomplete, ["tmax_sampled_c", "tmin_sampled_c"]] = np.nan
    out["missing_reason"] = np.where(incomplete, "incomplete_samples", "")

    ver = (
        v[v["model_version"].astype(str) != "unknown"]
        .groupby(["model_id", "init_time"], observed=True)["model_version"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "unknown")
        .rename("model_version")
        .reset_index()
    )
    out = out.merge(ver, on=["model_id", "init_time"], how="left")
    out["model_version"] = out["model_version"].fillna("unknown")

    out["lead_day"] = (
        (out["_day"] - out["init_time"].dt.floor("D")).dt.days.astype("int16").astype("int8")
    )
    out["climo_date"] = out["_day"].dt.date
    for c in ("tmax_sampled_c", "tmin_sampled_c", "tmax_native_c", "tmin_native_c"):
        out[c] = out[c].astype("float32")
    out["schema_version"] = SCHEMA_VERSION
    out["methodology_version"] = METHODOLOGY_VERSION
    out = out[DAILY_COLUMNS].sort_values(DAILY_COLUMNS[:6]).reset_index(drop=True)
    return out
