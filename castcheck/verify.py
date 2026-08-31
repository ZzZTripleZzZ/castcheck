"""Scores, bootstrap confidence intervals, pairwise comparisons and the persistence baseline.

Implements METHODOLOGY §4 (scores), §5 (uncertainty) and §6 (missing data), producing the two
published tables of DESIGN §3.4 (`scores`) and §3.5 (`pairwise`).

What is scored (v0.3, DESIGN §10.2)
-----------------------------------
The external review (`docs/06-external-review-v02.md`) showed that verifying a *sampled* daily
extreme against the *true* NWS extreme makes the headline number depend on each model's own diurnal
amplitude — the very thing the site was reporting.  v0.3 therefore scores several distinct
functionals and keeps them apart by name:

======================  ==========================================  =============================
`variable`              forecast                                    truth
======================  ==========================================  =============================
``t2``                  instantaneous 2 m T at the 4 common instants observation at the same instant
``t2_00z`` … ``t2_18z`` the same, one valid hour at a time          observation at the same instant
``tmax_s`` ``tmin_s``   max/min of the 4 forecast samples           max/min of the 4 *observed* samples
``tmax_cli``            max/min of the 4 forecast samples           NWS CLI daily extreme
``tmin_cli``
``tmax_native_cli``     the model's own window extreme (§2.4)       NWS CLI daily extreme
``tmin_native_cli``
======================  ==========================================  =============================

``t2`` is the headline: it is the only column with no extreme-sampling definition in it at all.
``tmax_s``/``tmin_s`` are like-for-like daily extremes.  The ``*_cli`` columns are secondary — they
answer "what does a daily-max user experience", and they carry a sampling penalty whose size depends
on the model's diurnal amplitude, so they must never be read as a pure forecast-error ranking.

Design notes
------------
*Truth selection* — the observation used for a station-day/variable is the first-final NWS CLI value;
if that is absent, CF6; if that is absent, the hourly-observation fallback, which is always flagged.
Instantaneous truth comes from ``truth_instant`` (DESIGN §10.1) and is joined upstream, in
:func:`castcheck.derive.instant_errors`.

*Bootstrap (v0.3: per group)* — METHODOLOGY §5 asks for 1000 resamples of scored **days**.  v0.2 drew
one resample-count matrix per *window* and shared it across all groups; review item A3 showed this
gives an unstable, group-size-dependent interval for sparse groups (four windows over the same 28
realised days produced four different intervals for the same point estimate).  v0.3 draws the
resample on **each group's own realised date axis**: circular moving blocks of
:data:`BLOCK_DAYS` = 7 days, ``n_boot`` resamples, percentile interval.  The interval is ``NaN`` when
the group has fewer than :data:`MIN_N_CI` days or fewer than :data:`MIN_BLOCKS` blocks to draw from.

Doing that group by group in Python would be ~400 000 loops.  Instead groups are bucketed by their
*realised date set*: every group with the same set of scored days gets the same resample matrix, and
all of its columns are evaluated in one ``(n_boot × n_days) @ (n_days × k)`` matrix product.  The
matrix is seeded from a stable hash of the date set, so the same date set produces the same
resample in every window and in every run — which is exactly the invariant A3 asked for: identical
data ⇒ identical interval, whatever window it is displayed in.

*Aggregate ("ALL") rows* — for ``station_id="ALL"`` the daily series is first averaged across the
stations that have a value on that day (separately for |error|, error, error², each hit indicator and
the out-of-sample debiased |error|), and the statistics and bootstrap are then computed over days
exactly as for a station.  Pooling all station-days instead would weight a day with 23 stations 23× a
day with one, and would break the exchangeability of the day as the resampling unit; ``n_stations``
records the mean number of stations behind an ALL row (METHODOLOGY §4).

*Model versions* — a model's scores are never aggregated across a cycle/weight change
(METHODOLOGY §7).  Each ``model_id`` is truncated to its most recent contiguous ``model_version``
segment before scoring; ``model_version`` and ``segment_start`` are published with every row.

Complexity
----------
Let ``D`` be the number of distinct climatological days, ``C`` the number of published columns
(station × init × lead × variable × method × model, including the ``ALL`` and persistence columns),
``W`` the number of windows, ``B`` the number of bootstrap resamples and ``P`` the number of model
pairs.  Building the error and unit tables is ``O(R log R)`` in the number of scored rows ``R``
(sorts and group-bys).  The point statistics are ``O(W · D · C)`` dense reductions.  The bootstrap is
``O(W · B · D̄ · C)`` BLAS flops (``D̄`` = mean realised days per group) plus ``O(W · B · C)`` for the
percentiles, and the same again for the skill intervals; pairwise adds ``O(W · B · D̄ · P)``.  The
date-set bucketing costs ``O(W · nnz)`` and turns what would be ``W · C`` small resamples into a few
dozen matrix products.  Everything is chunked over cells (``max_cells_per_chunk``) so peak memory is
``O(D · chunk_columns)``, not ``O(D · C)``.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION
from .climo_day import COMMON_SAMPLE_HOURS_UTC

__all__ = [
    "ALL_STATIONS",
    "BLOCK_DAYS",
    "DEBIAS_MIN_HISTORY",
    "DEBIAS_WINDOW",
    "MIN_BLOCKS",
    "MIN_N_CI",
    "PAIRWISE_COLUMNS",
    "PERSISTENCE_ID",
    "SCORE_COLUMNS",
    "VARIABLES",
    "error_table",
    "holm_reject",
    "latest_version_segments",
    "persistence_daily",
    "score",
    "select_truth",
    "wilson_interval",
]

PERSISTENCE_ID = "persistence"
F_TO_C = 5.0 / 9.0
HIT_THRESHOLDS_C = (1.0 * F_TO_C, 2.0 * F_TO_C, 3.0 * F_TO_C)
MIN_N = 30  # METHODOLOGY §4: windows below this are published but greyed out
ALL_STATIONS = "ALL"

#: Truth variables inside ``truth_daily`` (the CLI/CF6 daily extremes).
TRUTH_VARIABLES = ("tmax", "tmin")
#: Instant-level score variables (DESIGN §10.2).
INSTANT_VARIABLES = ("t2", "t2_00z", "t2_06z", "t2_12z", "t2_18z")
#: Daily score variables, in publication order.
DAILY_VARIABLES = ("tmax_s", "tmin_s", "tmax_cli", "tmin_cli", "tmax_native_cli", "tmin_native_cli")
VARIABLES = INSTANT_VARIABLES + DAILY_VARIABLES

#: Which forecast/truth column pair each daily variable is built from.
_DAILY_SPEC = {
    "tmax_s": ("tmax_sampled_c", "obs_sampled", "tmax_obs_s_c"),
    "tmin_s": ("tmin_sampled_c", "obs_sampled", "tmin_obs_s_c"),
    "tmax_cli": ("tmax_sampled_c", "cli", "tmax"),
    "tmin_cli": ("tmin_sampled_c", "cli", "tmin"),
    "tmax_native_cli": ("tmax_native_c", "cli", "tmax"),
    "tmin_native_cli": ("tmin_native_c", "cli", "tmin"),
}

#: Moving-block length (days) for the day bootstrap (METHODOLOGY §5).  One week: longer than
#: the synoptic decorrelation time of a daily 2 m temperature error.
BLOCK_DAYS = 7
#: DESIGN §10.3: no interval below this many scored days …
MIN_N_CI = 28
#: … and none with fewer than this many blocks to draw from.
MIN_BLOCKS = 4
#: Out-of-sample debiasing (DESIGN §10.3): trailing window and the minimum history it needs.
DEBIAS_WINDOW = 30
DEBIAS_MIN_HISTORY = 15
#: Two-sided level used for `distinguishable_*` and for every published interval.
ALPHA = 0.05
UNKNOWN_VERSION = "unknown"

SCORE_COLUMNS = [
    "station_id", "model_id", "init_hour", "lead_day", "variable", "method", "window",
    "n", "n_stations", "n_flagged", "mae", "bias", "rmse", "hit1f", "hit2f", "hit3f",
    "mae_debiased", "n_debiased",
    "n_common", "mae_persistence_common", "skill_persistence", "skill_persistence_debiased",
    "skill_ci_low", "skill_ci_high",
    "mae_ci_low", "mae_ci_high", "bias_ci_low", "bias_ci_high",
    "rmse_ci_low", "rmse_ci_high", "hit1f_ci_low", "hit1f_ci_high",
    "model_version", "segment_start",
    "period_start", "period_end", "computed_at", "methodology_version", "schema_version",
]

# DESIGN §3.5 columns, plus `method` (the comparison is only meaningful within one extraction
# method; see module docstring of site/build.py for how it is surfaced).
PAIRWISE_COLUMNS = [
    "station_id", "init_hour", "lead_day", "variable", "window", "model_a", "model_b",
    "n_common", "mae_diff", "ci_low", "ci_high", "p_boot",
    "distinguishable_uncorrected", "distinguishable_holm", "method",
    "computed_at", "methodology_version", "schema_version",
]

# METHODOLOGY §3: the truth is the *first final* CLI value.  A same-day preliminary CLI
# ("TODAY ... VALID AS OF") is never used, not even as a last resort — it is dropped, not ranked
# below the hourly-observation fallback.
_TRUTH_RANK = {("CLI", True): 0, ("CF6", False): 1, ("CF6", True): 1, ("OBS", False): 2,
               ("OBS", True): 2}

#: ``sub`` distinguishes the four rows a pooled ``t2`` group-day carries (it holds the valid hour);
#: it is 0 for every one-row-per-day variable and never leaves this module.
_ROW_COLS = ["station_id", "model_id", "init_hour", "lead_day", "variable", "method",
             "climo_date", "sub", "fcst_c", "obs_c", "qc_flag"]


def empty_scores() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SCORE_COLUMNS})


def empty_pairwise() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PAIRWISE_COLUMNS})


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _ROW_COLS})


def window_label(w: int | None) -> str:
    return "all" if w is None else f"{int(w)}d"


# --------------------------------------------------------------------------------------------
# truth
# --------------------------------------------------------------------------------------------

def select_truth(truth: pd.DataFrame) -> pd.DataFrame:
    """One observation per ``(station_id, climo_date, variable)`` with the METHODOLOGY §3 priority.

    Returns columns ``station_id, climo_date, variable, obs_c, truth_source, qc_flag`` where
    ``variable`` is ``tmax``/``tmin`` — the *truth* variable, not the score variable.
    """
    cols = ["station_id", "climo_date", "variable", "obs_c", "truth_source", "qc_flag"]
    if truth is None or len(truth) == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    t = truth.copy()
    t["climo_date"] = pd.to_datetime(t["climo_date"]).dt.normalize()
    if "is_final" not in t:
        t["is_final"] = False
    t["is_final"] = t["is_final"].fillna(False).astype(bool)
    if "qc_flag" not in t:
        t["qc_flag"] = ""
    t["qc_flag"] = t["qc_flag"].fillna("")
    t["_rank"] = [
        _TRUTH_RANK.get((str(s), bool(f)), 9) for s, f in zip(t["source"], t["is_final"])
    ]
    frames = []
    for var in TRUTH_VARIABLES:
        col = f"{var}_c"
        if col not in t:
            continue
        part = t[["station_id", "climo_date", "source", "qc_flag", "_rank"]].copy()
        part["obs_c"] = pd.to_numeric(t[col], errors="coerce")
        part["variable"] = var
        frames.append(part[part["obs_c"].notna()])
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    long = pd.concat(frames, ignore_index=True)
    long = long[long["_rank"] < 9]  # drop preliminary CLI and any unknown source
    if long.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    long = long.sort_values(["station_id", "climo_date", "variable", "_rank"])
    long = long.drop_duplicates(subset=["station_id", "climo_date", "variable"], keep="first")
    long = long.rename(columns={"source": "truth_source"})
    flag = long["qc_flag"].astype(str)
    long["qc_flag"] = np.where(
        (long["truth_source"] == "OBS") & (flag == ""), "obs_fallback", flag
    )
    return long[cols].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# the scored-row table: one row per (group, day) — four per day for the pooled `t2`
# --------------------------------------------------------------------------------------------

def _instant_rows(instant: pd.DataFrame) -> pd.DataFrame:
    """``t2`` (pooled over the four instants) and ``t2_00z…t2_18z`` rows."""
    if instant is None or len(instant) == 0:
        return _empty_rows()
    i = instant.copy()
    i["init_hour"] = pd.to_datetime(i["init_time"], utc=True).dt.hour.astype("int16")
    i["climo_date"] = pd.to_datetime(i["climo_date"]).dt.normalize()
    i["fcst_c"] = pd.to_numeric(i["fcst_c"], errors="coerce")
    i["obs_c"] = pd.to_numeric(i["obs_c"], errors="coerce")
    if "qc_flag" not in i:
        i["qc_flag"] = ""
    i["qc_flag"] = i["qc_flag"].fillna("").astype(str)
    i = i[i["fcst_c"].notna() & i["obs_c"].notna()]
    if i.empty:
        return _empty_rows()
    hours = i["valid_hour_utc"].astype("int16")
    base = i[["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date",
              "fcst_c", "obs_c", "qc_flag"]]
    pooled = base.assign(variable="t2", sub=hours)
    hourly = base.assign(
        variable=[f"t2_{int(h):02d}z" for h in hours], sub=np.int16(0)
    )
    return pd.concat([pooled[_ROW_COLS], hourly[_ROW_COLS]], ignore_index=True)


def _daily_rows(daily: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """``tmax_s/tmin_s`` (like-for-like sampled extremes) and the ``*_cli`` secondary variables."""
    if daily is None or len(daily) == 0:
        return _empty_rows()
    d = daily.copy()
    d["init_hour"] = pd.to_datetime(d["init_time"], utc=True).dt.hour.astype("int16")
    d["climo_date"] = pd.to_datetime(d["climo_date"]).dt.normalize()
    sel = select_truth(truth)
    cli = {}
    if len(sel):
        for var in TRUTH_VARIABLES:
            part = sel[sel["variable"] == var]
            cli[var] = part[["station_id", "climo_date", "obs_c", "qc_flag"]]

    key = ["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date"]
    frames = []
    for variable, (fcol, truth_kind, tcol) in _DAILY_SPEC.items():
        if fcol not in d.columns:
            continue
        part = d[key].copy()
        part["fcst_c"] = pd.to_numeric(d[fcol], errors="coerce")
        part = part[part["fcst_c"].notna()]
        if part.empty:
            continue
        if truth_kind == "obs_sampled":
            if tcol not in d.columns:
                continue
            part["obs_c"] = pd.to_numeric(d.loc[part.index, tcol], errors="coerce")
            part["qc_flag"] = ""
        else:
            obs = cli.get(tcol)
            if obs is None or obs.empty:
                continue
            part = part.merge(obs, on=["station_id", "climo_date"], how="inner")
        part = part[part["obs_c"].notna()]
        if part.empty:
            continue
        part["variable"] = variable
        part["sub"] = np.int16(0)
        part["qc_flag"] = part["qc_flag"].fillna("").astype(str)
        frames.append(part[_ROW_COLS])
    if not frames:
        return _empty_rows()
    return pd.concat(frames, ignore_index=True)


def _lagged(targets: pd.DataFrame, source: pd.DataFrame, on: list[str], time_col: str,
            lead: int) -> pd.DataFrame:
    """``targets`` joined to the observation ``lead`` days earlier — the persistence forecast."""
    src = source[[*on, time_col, "obs_c"]].rename(columns={"obs_c": "fcst_c"}).copy()
    src[time_col] = src[time_col] + pd.Timedelta(days=int(lead))
    return targets.merge(src, on=[*on, time_col], how="inner")


def _persistence_rows(
    rows: pd.DataFrame,
    instant: pd.DataFrame | None,
    truth: pd.DataFrame | None = None,
    truth_instant: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Lagged-persistence baseline rows for every variable (DESIGN §10.3).

    For target day ``D`` at lead day ``L`` the baseline is the observation ``L`` days earlier —
    *of the same functional*: the observation at the same UTC hour for the ``t2*`` variables, the
    observed sampled extreme for ``tmax_s/tmin_s``, the CLI extreme for the ``*_cli`` variables.
    v0.2 used the CLI daily extreme for everything, so the baseline carried no sampling penalty
    while the numerator did, and ``skill_persistence`` came out negative for every model at lead 1
    (review item B3).  With the like-for-like baseline the ratio is again a statement about forecast
    quality.

    The *targets* are the group-days actually scored; the *sources* come from the full observation
    record when it is supplied (``truth``, ``truth_instant``), so that a lead-9 baseline is not lost
    merely because the forecast archive does not reach nine days before its own first day.

    Rows are emitted for every ``(init_hour, method)`` present in the data so that the baseline lines
    up with every model group; the baseline itself depends on neither.
    """
    if rows is None or len(rows) == 0:
        return _empty_rows()
    leads = sorted({int(x) for x in pd.to_numeric(rows["lead_day"], errors="coerce").dropna()
                    if int(x) >= 1})
    if not leads:
        return _empty_rows()
    init_hours = sorted({int(x) for x in rows["init_hour"].unique()})
    methods = sorted(set(rows["method"].astype(str).unique()))

    out: list[pd.DataFrame] = []

    # --- instantaneous variables: the observation at the same UTC hour, `lead` days earlier ------
    if instant is not None and len(instant):
        i = instant.copy()
        i["valid_time"] = pd.to_datetime(i["valid_time"], utc=True)
        i["climo_date"] = pd.to_datetime(i["climo_date"]).dt.normalize()
        i["obs_c"] = pd.to_numeric(i["obs_c"], errors="coerce")
        targets = (
            i[i["obs_c"].notna()]
            [["station_id", "valid_time", "valid_hour_utc", "climo_date", "obs_c", "qc_flag"]]
            .drop_duplicates(subset=["station_id", "valid_time"])
        )
        if truth_instant is not None and len(truth_instant):
            src = truth_instant.copy()
            src["valid_time"] = pd.to_datetime(src["valid_time"], utc=True)
            src["obs_c"] = pd.to_numeric(src["temp_c"], errors="coerce")
            src = src[src["obs_c"].notna()][["station_id", "valid_time", "obs_c"]]
            src = src.drop_duplicates(subset=["station_id", "valid_time"])
        else:
            src = (
                i[i["obs_c"].notna()][["station_id", "valid_time", "obs_c"]]
                .drop_duplicates(subset=["station_id", "valid_time"])
            )
        for lead in leads:
            m = _lagged(targets, src, ["station_id"], "valid_time", lead)
            if m.empty:
                continue
            m["lead_day"] = int(lead)
            hours = m["valid_hour_utc"].astype("int16")
            out.append(pd.concat([
                m.assign(variable="t2", sub=hours),
                m.assign(variable=[f"t2_{int(h):02d}z" for h in hours], sub=np.int16(0)),
            ], ignore_index=True))

    # --- daily variables ------------------------------------------------------------------------
    targets = (
        rows[rows["variable"].isin(DAILY_VARIABLES)]
        [["station_id", "climo_date", "variable", "obs_c", "qc_flag"]]
        .drop_duplicates(subset=["station_id", "climo_date", "variable"])
    )
    if len(targets):
        src = _daily_observation_record(rows, truth, truth_instant)
        for lead in leads:
            m = _lagged(targets, src, ["station_id", "variable"], "climo_date", lead)
            if m.empty:
                continue
            m["lead_day"] = int(lead)
            m["sub"] = np.int16(0)
            out.append(m)

    if not out:
        return _empty_rows()
    base = pd.concat(out, ignore_index=True)
    base["model_id"] = PERSISTENCE_ID
    parts = []
    for init_hour, method in itertools.product(init_hours, methods):
        p = base.copy()
        p["init_hour"] = np.int16(init_hour)
        p["method"] = method
        parts.append(p[_ROW_COLS])
    return pd.concat(parts, ignore_index=True)


def _daily_observation_record(
    rows: pd.DataFrame, truth: pd.DataFrame | None, truth_instant: pd.DataFrame | None
) -> pd.DataFrame:
    """``station_id, climo_date, variable, obs_c`` for every daily variable, over the full record."""
    frames = []
    sel = select_truth(truth) if truth is not None else None
    if sel is not None and len(sel):
        for variable, (_f, kind, tcol) in _DAILY_SPEC.items():
            if kind != "cli":
                continue
            part = sel[sel["variable"] == tcol][["station_id", "climo_date", "obs_c"]].copy()
            part["variable"] = variable
            frames.append(part)
    if truth_instant is not None and len(truth_instant):
        from .derive import observed_sampled_extremes

        obs_s = observed_sampled_extremes(truth_instant)
        if len(obs_s):
            obs_s = obs_s.copy()
            obs_s["climo_date"] = pd.to_datetime(obs_s["climo_date"]).dt.normalize()
            for variable, col in (("tmax_s", "tmax_obs_s_c"), ("tmin_s", "tmin_obs_s_c")):
                part = obs_s[["station_id", "climo_date"]].copy()
                part["obs_c"] = pd.to_numeric(obs_s[col], errors="coerce")
                part["variable"] = variable
                frames.append(part[part["obs_c"].notna()])
    # lowest priority: whatever observation record the scored rows themselves carry.  It covers only
    # the days some model forecast, but it is the only source for a variable (or a station) the
    # registry-based lookups above could not resolve.
    frames.append(rows[rows["variable"].isin(DAILY_VARIABLES)]
                  [["station_id", "climo_date", "variable", "obs_c"]])
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["station_id", "climo_date", "variable", "obs_c"])
    rec = pd.concat(frames, ignore_index=True)
    rec["climo_date"] = pd.to_datetime(rec["climo_date"]).dt.normalize()
    return rec.drop_duplicates(subset=["station_id", "climo_date", "variable"])


def _complete_pooled_days(err: pd.DataFrame) -> pd.DataFrame:
    """Drop pooled-``t2`` group-days that do not have all four instants.

    The pooled ``t2`` score of a day is the mean over its instants, so a day with one usable instant
    would otherwise weigh as much as a day with four, and two models with different instant coverage
    would be averaging different things.  This is the instantaneous twin of the ``n_samples = 4``
    rule the sampled extremes already obey (METHODOLOGY §2.3).  The per-hour ``t2_00z…`` variables
    are unaffected: there the day *is* the instant.
    """
    if err.empty or "t2" not in set(err["variable"].unique()):
        return err
    is_pooled = err["variable"] == "t2"
    n_sub = err[is_pooled].groupby(_KEY_COLS + ["climo_date"], observed=True)["sub"].transform("size")
    keep = pd.Series(True, index=err.index)
    keep.loc[n_sub.index] = n_sub.to_numpy() == len(COMMON_SAMPLE_HOURS_UTC)
    return err[keep].reset_index(drop=True)


def error_table(daily: pd.DataFrame, truth: pd.DataFrame,
                instant: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every scored row: ``(group, day)`` with ``fcst_c``, ``obs_c`` and the signed error.

    The pooled ``t2`` variable has four rows per group-day (one per common instant); every other
    variable has one.
    """
    frames = [_daily_rows(daily, truth)]
    if instant is not None and len(instant):
        frames.append(_instant_rows(instant))
    rows = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(
        len(f) for f in frames) else _empty_rows()
    if rows.empty:
        return rows.assign(err=pd.Series(dtype="float64"))
    rows["err"] = rows["fcst_c"].astype("float64") - rows["obs_c"].astype("float64")
    rows = rows[rows["err"].notna()].reset_index(drop=True)
    return rows


# --------------------------------------------------------------------------------------------
# persistence baseline in the daily_forecasts schema (kept for the CLI / cross-check scripts)
# --------------------------------------------------------------------------------------------

def persistence_daily(
    truth: pd.DataFrame,
    leads: tuple[int, ...] = tuple(range(1, 10)),
    methods: tuple[str, ...] = ("bilinear", "nearest"),
    init_hours: tuple[int, ...] = (0, 12),
) -> pd.DataFrame:
    """Persistence baseline in the ``daily_forecasts`` schema, from the CLI daily extremes.

    **Deprecated as a scoring input.** :func:`score` builds its own baseline per variable (see
    :func:`_persistence_rows`) and *drops* any ``model_id == "persistence"`` rows it is handed, so
    passing this frame in is harmless but has no effect.  It survives because
    ``scripts/crosscheck_verify.py`` and the CLI still materialise a baseline in the daily schema.

    **Definition (lagged persistence).** For target climatological day ``D`` and lead day ``L`` the
    persistence forecast is the observation of day ``D − L`` — the last observation a forecaster
    issuing at lead ``L`` could already have seen.  ``L = 1`` is classic persistence; larger ``L``
    degrade it the way a model's skill degrades, so that ``skill = 1 − MAE_model/MAE_persistence``
    compares like with like at every lead.  Lead day 0 is not produced because ``D − 0`` is the
    target day itself.  The rejected alternative — "yesterday's observation at every lead" — would
    give the baseline information the forecast did not have.
    """
    from .derive import daily_columns

    cols = daily_columns()
    sel = select_truth(truth)
    if sel.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    wide = sel.pivot_table(
        index=["station_id", "climo_date"], columns="variable", values="obs_c", aggfunc="first"
    ).reset_index()
    for var in TRUTH_VARIABLES:
        if var not in wide.columns:
            wide[var] = np.nan
    wide.columns.name = None

    targets = wide[["station_id", "climo_date"]].copy()
    frames = []
    for lead in leads:
        src = wide.rename(columns={"tmax": "_tmax", "tmin": "_tmin"}).copy()
        src["climo_date"] = src["climo_date"] + pd.Timedelta(days=int(lead))
        m = targets.merge(src[["station_id", "climo_date", "_tmax", "_tmin"]],
                          on=["station_id", "climo_date"], how="inner")
        m = m[m["_tmax"].notna() | m["_tmin"].notna()]
        if m.empty:
            continue
        m["lead_day"] = int(lead)
        frames.append(m)
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    base = pd.concat(frames, ignore_index=True)

    out = []
    for init_hour in init_hours:
        for method in methods:
            part = base.copy()
            part["method"] = method
            part["init_time"] = (
                part["climo_date"] - pd.to_timedelta(part["lead_day"], unit="D")
                + pd.Timedelta(hours=int(init_hour))
            ).dt.tz_localize("UTC")
            out.append(part)
    df = pd.concat(out, ignore_index=True)
    df["model_id"] = PERSISTENCE_ID
    df["model_version"] = "obs"
    df["tmax_sampled_c"] = df["_tmax"].astype("float32")
    df["tmin_sampled_c"] = df["_tmin"].astype("float32")
    df["n_samples"] = np.int8(4)
    for c in ("tmax_native_c", "tmin_native_c", "tmax_obs_s_c", "tmin_obs_s_c",
              "native_overhang_h"):
        df[c] = np.float32(np.nan)
    df["n_obs_samples"] = np.int8(0)
    df["missing_reason"] = ""
    df["lead_day"] = df["lead_day"].astype("int8")
    df["climo_date"] = df["climo_date"].dt.date
    df["schema_version"] = SCHEMA_VERSION
    df["methodology_version"] = METHODOLOGY_VERSION
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# version segmentation
# --------------------------------------------------------------------------------------------

def latest_version_segments(daily: pd.DataFrame) -> pd.DataFrame:
    """Per ``model_id``: its most recent ``model_version`` and the first init time of that segment.

    METHODOLOGY §7 forbids aggregating scores across a cycle or weight change, so :func:`score`
    drops every row initialized before ``segment_start``.  ``"unknown"`` versions never open or
    close a segment: they inherit whatever known version brackets them.  Baselines (persistence) and
    models whose version never changed have ``segment_start`` equal to their first init.
    """
    cols = ["model_id", "model_version", "segment_start"]
    if daily is None or len(daily) == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    d = daily[["model_id", "model_version", "init_time"]].copy()
    d["model_version"] = d["model_version"].fillna(UNKNOWN_VERSION).astype(str)
    d["init_time"] = pd.to_datetime(d["init_time"], utc=True)
    out = []
    for model_id, part in d.groupby("model_id", observed=True):
        known = part[part["model_version"] != UNKNOWN_VERSION].sort_values("init_time")
        if known.empty:
            out.append({"model_id": model_id, "model_version": UNKNOWN_VERSION,
                        "segment_start": part["init_time"].min()})
            continue
        last = known["model_version"].iat[-1]
        # first init at or after which no *other* known version appears
        differs = np.nonzero(known["model_version"].to_numpy() != last)[0]
        start = known["init_time"].iat[int(differs.max()) + 1] if len(differs) else known["init_time"].iat[0]
        out.append({"model_id": model_id, "model_version": last, "segment_start": start})
    return pd.DataFrame(out)[cols]


# --------------------------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------------------------

_Z = 1.959963984540054  # two-sided 95 %


def wilson_interval(successes, n, z: float = _Z):
    """Wilson score interval for a proportion, vectorised, clipped to ``[0, 1]``.

    ``successes`` may be fractional: for the pooled ``t2`` variable and for ``ALL`` rows the daily
    hit indicator is itself a mean (over four instants, or over stations), so the "number of
    successes" is ``n × hit_rate``.  The interval then treats the *day* as the independent Bernoulli
    unit, which is the same unit the bootstrap uses; it is an approximation for those rows and an
    exact Wilson interval for a single-station, single-instant variable.  Its point of being here is
    that it never degenerates: a 0/28 hit rate reports ``[0, 0.12]``, not ``[0, 0]`` (review A3).
    """
    n = np.asarray(n, dtype="float64")
    k = np.asarray(successes, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
        lo = np.clip(centre - half, 0.0, 1.0)
        hi = np.clip(centre + half, 0.0, 1.0)
    bad = ~(n > 0)
    return np.where(bad, np.nan, lo), np.where(bad, np.nan, hi)


def holm_reject(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Holm–Bonferroni step-down rejections for one family of p-values."""
    p = np.asarray(p, dtype="float64")
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf), kind="stable")
    ranked = p[order]
    raw = np.minimum(1.0, (m - np.arange(m)) * ranked)
    adjusted = np.maximum.accumulate(np.where(np.isfinite(raw), raw, np.inf))
    out = np.zeros(m, dtype=bool)
    out[order] = adjusted <= alpha
    return out & np.isfinite(p)


def _block_counts(n_days: int, n_boot: int, block_days: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_boot, n_days)`` circular moving-block resample counts on the group's own date axis.

    ``ceil(n_days / L)`` blocks of ``L`` consecutive days are drawn from uniformly random starts and
    wrapped around the axis; the concatenation is truncated to ``n_days``, so every row sums to
    exactly ``n_days`` and the bootstrap mean needs no self-normalisation.  ``block_days <= 1``
    degenerates to the plain multinomial (iid day) resample.
    """
    if block_days <= 1:
        return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot).astype("float32")
    L = int(min(block_days, n_days))
    n_blocks = int(np.ceil(n_days / L))
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n_days
    idx = idx.reshape(n_boot, -1)[:, :n_days]
    flat = idx + (np.arange(n_boot, dtype=idx.dtype) * n_days)[:, None]
    W = np.bincount(flat.ravel(), minlength=n_boot * n_days).astype("float32")
    return W.reshape(n_boot, n_days)


def _rng_for(seed: int, key: bytes) -> np.random.Generator:
    """A generator determined by ``seed`` and the group's realised date set.

    Because the seed is a function of the *dates* and not of the window or of the position in the
    chunk loop, two groups with the same scored days draw the same resample — in the 30-day window
    and in the 365-day window, in this run and in the next (review item A3).
    """
    h = hashlib.blake2b(key, digest_size=8, key=b"castcheck-boot").digest()
    return np.random.default_rng([int(seed), int.from_bytes(h, "little")])


def _pattern_groups(mask: np.ndarray, d0: int) -> dict[bytes, tuple[np.ndarray, np.ndarray]]:
    """Bucket columns of ``mask`` by their realised row set.

    ``mask`` is ``(n_window_days, k)`` boolean; ``d0`` is the window's offset into the global date
    axis, added so that the key identifies the *dates*, not their position inside the window.
    Returns ``key -> (window-local row indices, column indices)``.
    """
    k = mask.shape[1]
    cols, rows = np.nonzero(mask.T)
    if len(cols) == 0:
        return {}
    counts = np.bincount(cols, minlength=k)
    offs = np.concatenate([[0], np.cumsum(counts)])
    buckets: dict[bytes, list[int]] = defaultdict(list)
    keys: list[bytes | None] = [None] * k
    for j in range(k):
        if counts[j] == 0:
            continue
        seg = rows[offs[j]:offs[j + 1]]
        key = (seg + d0).astype("int32").tobytes()
        keys[j] = key
        buckets[key].append(j)
    out: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for key, js in buckets.items():
        j0 = js[0]
        out[key] = (rows[offs[j0]:offs[j0 + 1]].copy(), np.asarray(js, dtype="int64"))
    return out


def _percentiles(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5/97.5 percentiles down the bootstrap axis (NaN-safe, but fast when there is no NaN)."""
    with np.errstate(invalid="ignore"):
        fn = np.nanpercentile if np.isnan(boot).any() else np.percentile
        lo, hi = fn(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=0)
    return lo, hi


def _ci_ok(n_days: int, block_days: int) -> bool:
    """DESIGN §10.3: publish an interval only for a group with enough days *and* enough blocks."""
    if n_days < MIN_N_CI:
        return False
    L = max(1, min(int(block_days), n_days))
    return int(np.ceil(n_days / L)) >= MIN_BLOCKS


# --------------------------------------------------------------------------------------------
# the scoring machinery
# --------------------------------------------------------------------------------------------

_KEY_COLS = ["station_id", "model_id", "init_hour", "lead_day", "variable", "method"]

#: Per (group, day) functionals carried through the dense matrices.  ``ns`` is the number of
#: stations behind the unit (1 for a station row, the count for an ALL row), ``fl`` marks a day
#: whose truth carries a ``qc_flag`` (METHODOLOGY §6), ``ad`` is the out-of-sample debiased absolute
#: error and ``dm`` marks the days on which ``ad`` is defined.
_UNIT_COLS = ("a", "s", "q", "h1", "h2", "h3", "ns", "fl", "ad", "dm")


def _out_of_sample_debias(unit: pd.DataFrame) -> pd.DataFrame:
    """Add ``ad``/``dm``: |e − b̂| where b̂ is the mean error of the trailing scored days.

    ``b̂`` for day *i* of a series is the mean signed error over the previous
    :data:`DEBIAS_WINDOW` *scored* days of that same series, and is undefined (``dm = 0``) until
    :data:`DEBIAS_MIN_HISTORY` of them exist.  Nothing from day *i* or later enters ``b̂``, which is
    the whole point: the v0.2 ``mae_debiased`` fitted one constant on the same days it then scored,
    and review item B1 showed that this alone moved Aurora's skill from −0.92 to +0.48.

    Implemented as a segmented prefix sum over the series sorted by ``(group, day)``: O(rows).
    """
    unit = unit.sort_values(_KEY_COLS + ["climo_date"], kind="stable").reset_index(drop=True)
    gid = unit.groupby(_KEY_COLS, observed=True, sort=False).ngroup().to_numpy()
    n = len(unit)
    if n == 0:
        unit["ad"] = np.zeros(0, dtype="float64")
        unit["dm"] = np.zeros(0, dtype="float64")
        return unit
    counts = np.bincount(gid)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    s = unit["s"].to_numpy(dtype="float64")
    csum = np.cumsum(s)
    base = np.repeat(np.where(starts > 0, csum[np.maximum(starts - 1, 0)], 0.0), counts)
    incl = csum - base                       # sum over local indices [0, i]
    excl = incl - s                          # sum over local indices [0, i-1]
    pos = np.arange(n)
    local = pos - np.repeat(starts, counts)
    back = np.minimum(local, DEBIAS_WINDOW)
    src = pos - back
    hist_sum = excl - excl[src]
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = np.where(back >= DEBIAS_MIN_HISTORY, hist_sum / np.maximum(back, 1), np.nan)
    ad = np.abs(s - bias)
    unit["ad"] = np.nan_to_num(ad, nan=0.0)
    unit["dm"] = np.isfinite(ad).astype("float64")
    return unit


def _unit_frame(err: pd.DataFrame) -> pd.DataFrame:
    """Per (group, day) error functionals, including the cross-station ``ALL`` rows."""
    e = err
    a = e["err"].abs().to_numpy(dtype="float64")
    s = e["err"].to_numpy(dtype="float64")
    raw = e[_KEY_COLS + ["climo_date"]].copy()
    raw["a"] = a
    raw["s"] = s
    raw["q"] = s * s
    for i, thr in enumerate(HIT_THRESHOLDS_C, start=1):
        # METHODOLOGY §4: the threshold is inclusive; |err| == 1 °F counts as a hit.  The epsilon
        # absorbs the float32 → float64 round-trip of a value that is exactly 1 °F in °C.
        raw[f"h{i}"] = (a <= thr + 1e-9).astype("float64")
    flag = e["qc_flag"].astype(str) if "qc_flag" in e else pd.Series("", index=e.index)
    raw["fl"] = (flag.fillna("").to_numpy() != "").astype("float64")

    # the pooled `t2` variable carries four rows per group-day; every other variable carries one.
    # Collapsing by the mean of each functional is the §4 rule applied over instants instead of
    # stations, and is a no-op wherever there is only one row.
    if raw.duplicated(subset=_KEY_COLS + ["climo_date"]).any():
        unit = (
            raw.groupby(_KEY_COLS + ["climo_date"], observed=True, sort=False)
            .agg(a=("a", "mean"), s=("s", "mean"), q=("q", "mean"),
                 h1=("h1", "mean"), h2=("h2", "mean"), h3=("h3", "mean"), fl=("fl", "max"))
            .reset_index()
        )
    else:
        unit = raw
    unit["ns"] = 1.0
    unit = _out_of_sample_debias(unit)

    grp = ["model_id", "init_hour", "lead_day", "variable", "method", "climo_date"]
    allrows = (
        unit.groupby(grp, observed=True)
        .agg(
            a=("a", "mean"), s=("s", "mean"), q=("q", "mean"),
            h1=("h1", "mean"), h2=("h2", "mean"), h3=("h3", "mean"),
            ns=("ns", "sum"),
            fl=("fl", "max"),  # a day is flagged for ALL as soon as any of its stations is
            ad=("ad", "sum"), dm=("dm", "sum"),
        )
        .reset_index()
    )
    # ALL's debiased error is the cross-station mean over the stations where it is defined
    with np.errstate(invalid="ignore", divide="ignore"):
        allrows["ad"] = np.where(allrows["dm"] > 0, allrows["ad"] / allrows["dm"], 0.0)
    allrows["dm"] = (allrows["dm"] > 0).astype("float64")
    allrows["station_id"] = ALL_STATIONS
    return pd.concat(
        [unit[_KEY_COLS + ["climo_date"] + list(_UNIT_COLS)],
         allrows[_KEY_COLS + ["climo_date"] + list(_UNIT_COLS)]],
        ignore_index=True,
    )


def _codes(unit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Factorise groups into dense column ids ordered so that all models of a cell are adjacent."""
    cell_keys = ["station_id", "init_hour", "lead_day", "variable", "method"]
    cell_df = unit[cell_keys].drop_duplicates().sort_values(cell_keys).reset_index(drop=True)
    cell_df["cell_id"] = np.arange(len(cell_df), dtype="int64")
    unit = unit.merge(cell_df, on=cell_keys, how="left")
    col_df = (
        unit[["cell_id", "model_id"]].drop_duplicates().sort_values(["cell_id", "model_id"])
        .reset_index(drop=True)
    )
    col_df["col_id"] = np.arange(len(col_df), dtype="int64")
    unit = unit.merge(col_df, on=["cell_id", "model_id"], how="left")
    col_df = col_df.merge(cell_df, on="cell_id", how="left")
    return unit["col_id"].to_numpy(), unit["cell_id"].to_numpy(), col_df, cell_df


def score(
    daily: pd.DataFrame,
    truth: pd.DataFrame,
    instant: pd.DataFrame | None = None,
    windows: tuple[int | None, ...] = (30, 90, 365, None),
    n_boot: int = 1000,
    seed: int = 0,
    as_of: date | str | None = None,
    pairwise_methods: tuple[str, ...] | None = None,
    max_cells_per_chunk: int = 400,
    block_days: int = BLOCK_DAYS,
    truth_instant: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the published ``scores`` and ``pairwise`` tables.

    ``instant`` is the instantaneous error table from :func:`castcheck.derive.instant_errors`; when
    it is ``None`` the ``t2*`` variables are simply absent from the output and only the daily
    variables are scored.  ``truth_instant`` (DESIGN §10.1) is optional and only widens the
    persistence baseline's source record beyond the days the forecast archive covers.
    ``pairwise_methods`` restricts the (expensive) model-vs-model comparison to a subset of
    extraction methods; ``None`` means all of them.  ``n_boot=0`` skips the bootstrap.
    """
    computed_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    have_daily = daily is not None and len(daily) > 0
    have_instant = instant is not None and len(instant) > 0
    if not have_daily and not have_instant:
        return empty_scores(), empty_pairwise()

    daily = daily.copy() if have_daily else None
    if daily is not None:
        # v0.3 builds its own per-variable baseline; anything handed in is dropped rather than
        # silently scored against the wrong truth (the v0.2 CLI baseline).
        daily = daily[daily["model_id"].astype(str) != PERSISTENCE_ID].reset_index(drop=True)

    # METHODOLOGY §7: never aggregate across a cycle / weight change
    seg_source = []
    if daily is not None and len(daily):
        seg_source.append(daily[["model_id", "model_version", "init_time"]])
    if have_instant:
        seg_source.append(instant[["model_id", "model_version", "init_time"]])
    segments = latest_version_segments(
        pd.concat(seg_source, ignore_index=True) if seg_source else None
    )
    if len(segments):
        seg_map = segments.set_index("model_id")["segment_start"]
        if daily is not None and len(daily):
            start = pd.to_datetime(daily["model_id"].map(seg_map), utc=True)
            keep = pd.to_datetime(daily["init_time"], utc=True) >= start
            daily = daily[keep.fillna(True)].reset_index(drop=True)
        if have_instant:
            start = pd.to_datetime(instant["model_id"].map(seg_map), utc=True)
            keep = pd.to_datetime(instant["init_time"], utc=True) >= start
            instant = instant[keep.fillna(True)].reset_index(drop=True)
            have_instant = len(instant) > 0

    err = error_table(daily, truth, instant if have_instant else None)
    if err.empty:
        return empty_scores(), empty_pairwise()
    pers = _persistence_rows(err, instant if have_instant else None, truth, truth_instant)
    if len(pers):
        pers["err"] = pers["fcst_c"].astype("float64") - pers["obs_c"].astype("float64")
        pers = pers[pers["err"].notna()]
        err = pd.concat([err, pers[[*_ROW_COLS, "err"]]], ignore_index=True)
    # DESIGN §3.3 already guarantees uniqueness; be defensive against duplicated shards
    err = err.drop_duplicates(subset=_KEY_COLS + ["climo_date", "sub"])
    err = _complete_pooled_days(err)

    if as_of is None:
        as_of_ts = pd.to_datetime(err["climo_date"]).max().normalize()
    else:
        as_of_ts = pd.Timestamp(as_of).normalize()
    as_of_ts = pd.Timestamp(as_of_ts).tz_localize(None) if getattr(as_of_ts, "tz", None) else as_of_ts

    unit = _unit_frame(err)
    col_of_row, _cell_of_row, col_df, cell_df = _codes(unit)

    dates = np.sort(unit["climo_date"].unique())
    date_index = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
    date_of_row = date_index.reindex(pd.DatetimeIndex(unit["climo_date"])).to_numpy()
    n_dates = len(dates)
    dates_ts = pd.DatetimeIndex(dates)

    vals = {k: unit[k].to_numpy(dtype="float32") for k in _UNIT_COLS}

    order = np.argsort(col_of_row, kind="stable")
    col_sorted = col_of_row[order]
    date_sorted = date_of_row[order].astype("int64")
    vals = {k: v[order] for k, v in vals.items()}
    n_cols = len(col_df)
    counts = np.bincount(col_sorted, minlength=n_cols)
    starts = np.concatenate([[0], np.cumsum(counts)])

    win_defs: list[tuple[str, int]] = []
    for w in windows:
        if w is None:
            d0 = 0
        else:
            cutoff = as_of_ts - pd.Timedelta(days=int(w) - 1)
            d0 = int(np.searchsorted(dates_ts.values, np.datetime64(cutoff), side="left"))
        if n_dates - d0 <= 0:
            continue
        win_defs.append((window_label(w), d0))

    pers_col = {}
    for cell_id, model_id, col_id in zip(col_df["cell_id"], col_df["model_id"], col_df["col_id"]):
        if model_id == PERSISTENCE_ID:
            pers_col[int(cell_id)] = int(col_id)

    cell_of_col = col_df["cell_id"].to_numpy()
    model_of_col = col_df["model_id"].to_numpy()
    want_pairwise = pairwise_methods is None or len(pairwise_methods) > 0
    pw_ok_cell = None
    if want_pairwise and pairwise_methods is not None:
        pw_ok_cell = set(cell_df.loc[cell_df["method"].isin(pairwise_methods), "cell_id"].tolist())

    score_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    do_boot = bool(n_boot)

    # chunk over cells so that every model of a cell lives in the same dense block
    cell_ids = cell_df["cell_id"].to_numpy()
    for c0 in range(0, len(cell_ids), max_cells_per_chunk):
        chunk_cells = cell_ids[c0: c0 + max_cells_per_chunk]
        lo_col = int(np.searchsorted(cell_of_col, chunk_cells[0], side="left"))
        hi_col = int(np.searchsorted(cell_of_col, chunk_cells[-1], side="right"))
        gc = hi_col - lo_col
        if gc == 0:
            continue
        r0, r1 = int(starts[lo_col]), int(starts[hi_col])
        rows_d = date_sorted[r0:r1]
        rows_c = col_sorted[r0:r1] - lo_col

        dense = {}
        for k in _UNIT_COLS:
            arr = np.zeros((n_dates, gc), dtype="float32")
            arr[rows_d, rows_c] = vals[k][r0:r1]
            dense[k] = arr
        M = np.zeros((n_dates, gc), dtype="float32")
        M[rows_d, rows_c] = 1.0

        p_local = np.array(
            [pers_col.get(int(cell_of_col[lo_col + g]), -1) for g in range(gc)], dtype="int64"
        )
        has_pers = p_local >= 0
        p_local_idx = np.where(has_pers, p_local - lo_col, 0)

        for label, d0 in win_defs:
            Mw = M[d0:]
            n = Mw.sum(axis=0)
            live = n > 0
            if not live.any():
                continue
            Aw, Sw, Qw = dense["a"][d0:], dense["s"][d0:], dense["q"][d0:]
            ADw, DMw = dense["ad"][d0:], dense["dm"][d0:]
            is_pers = model_of_col[lo_col:hi_col] == PERSISTENCE_ID
            with np.errstate(invalid="ignore", divide="ignore"):
                mae = Aw.sum(axis=0) / n
                bias = Sw.sum(axis=0) / n
                rmse = np.sqrt(Qw.sum(axis=0) / n)
                hits = [dense[f"h{i}"][d0:].sum(axis=0) / n for i in (1, 2, 3)]
                n_stations = dense["ns"][d0:].sum(axis=0) / n
                n_flagged = dense["fl"][d0:].sum(axis=0)
                n_debiased = DMw.sum(axis=0)
                mae_db = ADw.sum(axis=0) / n_debiased

                common = Mw * Mw[:, p_local_idx]
                nc = common.sum(axis=0)
                mae_self = (Aw * common).sum(axis=0) / nc
                mae_pers = (Aw[:, p_local_idx] * common).sum(axis=0) / nc
                skill = 1.0 - mae_self / mae_pers
                # the debiased twin, on the days where both series have an out-of-sample bias
                dcommon = DMw * DMw[:, p_local_idx]
                ndc = dcommon.sum(axis=0)
                db_self = (ADw * dcommon).sum(axis=0) / ndc
                db_pers = (ADw[:, p_local_idx] * dcommon).sum(axis=0) / ndc
                skill_db = 1.0 - db_self / db_pers
            ok_pers = has_pers & (nc > 0) & ~is_pers
            skill = np.where(ok_pers & (mae_pers > 0), skill, np.nan)
            skill_db = np.where(ok_pers & (ndc > 0) & (db_pers > 0), skill_db, np.nan)
            mae_pers_pub = np.where(ok_pers & (nc > 0), mae_pers, np.nan)
            n_common = np.where(ok_pers, nc, 0.0)
            mae_db = np.where(n_debiased > 0, mae_db, np.nan)

            nanv = np.full(gc, np.nan, dtype="float64")
            mae_lo, mae_hi = nanv.copy(), nanv.copy()
            bias_lo, bias_hi = nanv.copy(), nanv.copy()
            rmse_lo, rmse_hi = nanv.copy(), nanv.copy()
            skill_lo, skill_hi = nanv.copy(), nanv.copy()
            if do_boot:
                for key, (rows_idx, cols_idx) in _pattern_groups(Mw > 0, d0).items():
                    ng = len(rows_idx)
                    if not _ci_ok(ng, block_days):
                        continue
                    W = _block_counts(ng, n_boot, block_days, _rng_for(seed, key))
                    for j0 in range(0, len(cols_idx), _MAX_BOOT_COLS):
                        cj = cols_idx[j0: j0 + _MAX_BOOT_COLS]
                        V = np.concatenate(
                            [Aw[np.ix_(rows_idx, cj)], Sw[np.ix_(rows_idx, cj)],
                             Qw[np.ix_(rows_idx, cj)]], axis=1
                        )
                        bm = (W @ V) / ng
                        k = len(cj)
                        mae_lo[cj], mae_hi[cj] = _percentiles(bm[:, :k])
                        bias_lo[cj], bias_hi[cj] = _percentiles(bm[:, k: 2 * k])
                        rmse_lo[cj], rmse_hi[cj] = _percentiles(np.sqrt(bm[:, 2 * k: 3 * k]))
                        del bm, V
                # skill intervals live on the model∩persistence day set, which is a different
                # pattern from the model's own days whenever coverage differs.
                skill_mask = (common > 0) & ok_pers[None, :]
                for key, (rows_idx, cols_idx) in _pattern_groups(skill_mask, d0).items():
                    ng = len(rows_idx)
                    if not _ci_ok(ng, block_days):
                        continue
                    W = _block_counts(ng, n_boot, block_days, _rng_for(seed, key))
                    for j0 in range(0, len(cols_idx), _MAX_BOOT_COLS):
                        cj = cols_idx[j0: j0 + _MAX_BOOT_COLS]
                        num = W @ Aw[np.ix_(rows_idx, cj)]
                        den = W @ Aw[np.ix_(rows_idx, p_local_idx[cj])]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            sk = 1.0 - num / np.where(den > 0, den, np.nan)
                        skill_lo[cj], skill_hi[cj] = _percentiles(sk)
                        del num, den, sk

            hit1_lo, hit1_hi = wilson_interval(hits[0] * n, n)

            day_present = Mw > 0
            first_idx = np.argmax(day_present, axis=0)
            last_idx = n_dates - d0 - 1 - np.argmax(day_present[::-1], axis=0)
            w_dates = dates_ts[d0:]
            period_start = w_dates[first_idx].date
            period_end = w_dates[last_idx].date

            part = pd.DataFrame({
                "station_id": col_df["station_id"].to_numpy()[lo_col:hi_col],
                "model_id": model_of_col[lo_col:hi_col],
                "init_hour": col_df["init_hour"].to_numpy()[lo_col:hi_col],
                "lead_day": col_df["lead_day"].to_numpy()[lo_col:hi_col],
                "variable": col_df["variable"].to_numpy()[lo_col:hi_col],
                "method": col_df["method"].to_numpy()[lo_col:hi_col],
                "window": label,
                "n": n.astype("int32"),
                "n_stations": n_stations,
                "n_flagged": n_flagged.astype("int32"),
                "mae": mae, "bias": bias, "rmse": rmse,
                "hit1f": hits[0], "hit2f": hits[1], "hit3f": hits[2],
                "mae_debiased": mae_db, "n_debiased": n_debiased.astype("int32"),
                "n_common": n_common.astype("int32"),
                "mae_persistence_common": mae_pers_pub,
                "skill_persistence": skill,
                "skill_persistence_debiased": skill_db,
                "skill_ci_low": skill_lo, "skill_ci_high": skill_hi,
                "mae_ci_low": mae_lo, "mae_ci_high": mae_hi,
                "bias_ci_low": bias_lo, "bias_ci_high": bias_hi,
                "rmse_ci_low": rmse_lo, "rmse_ci_high": rmse_hi,
                "hit1f_ci_low": hit1_lo, "hit1f_ci_high": hit1_hi,
                "period_start": period_start, "period_end": period_end,
            })
            score_parts.append(part[live].reset_index(drop=True))

            if want_pairwise:
                pair_parts.extend(
                    _pairwise_chunk(
                        label=label, d0=d0, Aw=Aw, Mw=Mw, lo_col=lo_col, hi_col=hi_col,
                        cell_of_col=cell_of_col, model_of_col=model_of_col, col_df=col_df,
                        ok_cells=pw_ok_cell, n_boot=n_boot if do_boot else 0,
                        block_days=block_days, seed=seed,
                    )
                )
        del dense, M

    scores = pd.concat(score_parts, ignore_index=True) if score_parts else empty_scores()
    if len(scores):
        if len(segments):
            seg = segments.copy()
            seg["segment_start"] = pd.to_datetime(seg["segment_start"], utc=True).dt.date
            scores = scores.merge(seg, on="model_id", how="left")
        else:
            scores["model_version"] = UNKNOWN_VERSION
            scores["segment_start"] = pd.NaT
        scores.loc[scores["model_id"] == PERSISTENCE_ID, "model_version"] = "obs"
        scores["model_version"] = scores["model_version"].fillna(UNKNOWN_VERSION)
        scores["computed_at"] = computed_at
        scores["methodology_version"] = METHODOLOGY_VERSION
        scores["schema_version"] = SCHEMA_VERSION
        scores = scores[SCORE_COLUMNS]
        scores = scores.sort_values(
            ["station_id", "variable", "window", "lead_day", "init_hour", "method", "mae"]
        ).reset_index(drop=True)

    pairwise = pd.concat(pair_parts, ignore_index=True) if pair_parts else empty_pairwise()
    if len(pairwise):
        pairwise = _apply_holm(pairwise, scores)
        pairwise["computed_at"] = computed_at
        pairwise["methodology_version"] = METHODOLOGY_VERSION
        pairwise["schema_version"] = SCHEMA_VERSION
        pairwise = pairwise[PAIRWISE_COLUMNS].reset_index(drop=True)
    return scores, pairwise


#: Bootstrap matrices are materialised ``n_boot × k``; chunk ``k`` so that stays a few tens of MB.
_MAX_BOOT_COLS = 1500


def _pairwise_chunk(
    *, label, d0, Aw, Mw, lo_col, hi_col, cell_of_col, model_of_col, col_df, ok_cells,
    n_boot, block_days, seed, max_pair_cols: int = 4000,
) -> list[pd.DataFrame]:
    """Paired-bootstrap MAE differences for every model pair inside each cell of the chunk."""
    ia_all: list[int] = []
    ib_all: list[int] = []
    for cell in np.unique(cell_of_col[lo_col:hi_col]):
        if ok_cells is not None and int(cell) not in ok_cells:
            continue
        idx = np.nonzero(cell_of_col[lo_col:hi_col] == cell)[0]
        if len(idx) < 2:
            continue
        for a, b in itertools.combinations(idx.tolist(), 2):
            ia_all.append(a)
            ib_all.append(b)
    if not ia_all:
        return []
    ia = np.asarray(ia_all)
    ib = np.asarray(ib_all)
    out: list[pd.DataFrame] = []
    station = col_df["station_id"].to_numpy()[lo_col:hi_col]
    init_hour = col_df["init_hour"].to_numpy()[lo_col:hi_col]
    lead = col_df["lead_day"].to_numpy()[lo_col:hi_col]
    variable = col_df["variable"].to_numpy()[lo_col:hi_col]
    method = col_df["method"].to_numpy()[lo_col:hi_col]
    models = model_of_col[lo_col:hi_col]
    for p0 in range(0, len(ia), max_pair_cols):
        sa, sb = ia[p0: p0 + max_pair_cols], ib[p0: p0 + max_pair_cols]
        common = (Mw[:, sa] > 0) & (Mw[:, sb] > 0)
        nc = common.sum(axis=0)
        keep = nc > 0
        if not keep.any():
            continue
        d_day = (Aw[:, sa] - Aw[:, sb]) * common
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = d_day.sum(axis=0) / np.where(nc > 0, nc, np.nan)
        k = len(sa)
        lo = np.full(k, np.nan)
        hi = np.full(k, np.nan)
        pval = np.full(k, np.nan)
        if n_boot:
            for key, (rows_idx, cols_idx) in _pattern_groups(common, d0).items():
                ng = len(rows_idx)
                if not _ci_ok(ng, block_days):
                    continue
                W = _block_counts(ng, n_boot, block_days, _rng_for(seed, key))
                for j0 in range(0, len(cols_idx), _MAX_BOOT_COLS):
                    cj = cols_idx[j0: j0 + _MAX_BOOT_COLS]
                    bm = (W @ (Aw[np.ix_(rows_idx, sa[cj])] - Aw[np.ix_(rows_idx, sb[cj])])) / ng
                    lo[cj], hi[cj] = _percentiles(bm)
                    # two-sided bootstrap p: how far the resample distribution sits from zero
                    le = (bm <= 0).sum(axis=0)
                    ge = (bm >= 0).sum(axis=0)
                    pval[cj] = np.minimum(
                        1.0, 2.0 * np.minimum(le + 1, ge + 1) / (n_boot + 1)
                    )
                    del bm, le, ge
        sig = np.isfinite(lo) & np.isfinite(hi) & ((lo > 0) | (hi < 0))
        out.append(pd.DataFrame({
            "station_id": station[sa], "init_hour": init_hour[sa], "lead_day": lead[sa],
            "variable": variable[sa], "window": label, "model_a": models[sa],
            "model_b": models[sb], "n_common": nc.astype("int32"), "mae_diff": diff,
            "ci_low": lo, "ci_high": hi, "p_boot": pval,
            "distinguishable_uncorrected": sig,
            "distinguishable_holm": False,
            "method": method[sa],
        })[keep].reset_index(drop=True))
    return out


_TABLE_KEYS = ["station_id", "init_hour", "lead_day", "variable", "method", "window"]


def _apply_holm(pairwise: pd.DataFrame, scores: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Holm correction inside each displayed table, over the comparisons against its leader.

    The family is what a reader actually looks at: within one
    ``(station, init, lead, variable, method, window)`` table, every comparison involving the
    lowest-MAE model.  Rows outside that family keep ``distinguishable_holm = False`` — the site
    marks ▼/▲ only on the leader column, so those rows are never used to claim a difference.
    Baselines are excluded when choosing the leader but are compared against it like any other row.
    """
    pairwise = pairwise.copy()
    pairwise["distinguishable_holm"] = False
    if scores is None or not len(scores):
        return pairwise
    cand = scores[(scores["model_id"] != PERSISTENCE_ID) & scores["mae"].notna()]
    if cand.empty:
        return pairwise
    leaders = (
        cand.sort_values(_TABLE_KEYS + ["mae", "model_id"])
        .drop_duplicates(subset=_TABLE_KEYS, keep="first")[[*_TABLE_KEYS, "model_id"]]
        .rename(columns={"model_id": "_leader"})
    )
    pw = pairwise.merge(leaders, on=_TABLE_KEYS, how="left")
    pw.index = pairwise.index  # a left merge on a unique right key preserves row order
    fam = pw["_leader"].notna() & (
        (pw["model_a"] == pw["_leader"]) | (pw["model_b"] == pw["_leader"])
    ) & pw["p_boot"].notna()
    if not fam.any():
        return pairwise
    sub = pw[fam].copy()
    sub = sub.sort_values([*_TABLE_KEYS, "p_boot"], kind="stable")
    g = sub.groupby(_TABLE_KEYS, observed=True, sort=False)
    m = g["p_boot"].transform("size").to_numpy(dtype="float64")
    r = g.cumcount().to_numpy(dtype="float64")
    raw = np.minimum(1.0, (m - r) * sub["p_boot"].to_numpy(dtype="float64"))
    sub["_adj"] = pd.Series(raw, index=sub.index).groupby(
        [sub[c] for c in _TABLE_KEYS], observed=True, sort=False
    ).cummax()
    pairwise.loc[sub.index[sub["_adj"] <= alpha], "distinguishable_holm"] = True
    return pairwise
