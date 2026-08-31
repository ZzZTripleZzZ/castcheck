"""Scores, bootstrap confidence intervals, pairwise comparisons and the persistence baseline.

Implements METHODOLOGY §4 (scores), §5 (uncertainty) and §6 (missing data), producing the two
published tables of DESIGN §3.4 (`scores`) and §3.5 (`pairwise`).

Design notes
------------
*Truth selection* — the observation used for a station-day/variable is the first-final NWS CLI value;
if that is absent, CF6; if that is absent, the hourly-observation fallback, which is always flagged.

*Bootstrap* — METHODOLOGY §5 asks for 1000 resamples of scored **days**.  Doing that group-by-group
would mean ~200 000 independent resampling loops.  Instead one day-resample count matrix
``W`` (``n_boot × n_days``) is drawn per window and shared by every group, and each group's statistic
is evaluated as a *self-normalised weighted mean* over the days it actually has:

    mae*_b = Σ_d W[b,d]·|e_d|·m_d / Σ_d W[b,d]·m_d

Conditional on its realised size the restricted weight vector is exactly a resample of that group's
own days, so this is the standard day bootstrap with a randomised resample size; the ratio form
removes the first-order effect of that randomisation.  ``scripts/crosscheck_bootstrap.py`` measures
the residual: on the real archive the shared-``W`` interval is 0.99–1.06× the width of a textbook
per-group percentile bootstrap (median 1.015), which is within Monte-Carlo noise.  Sharing ``W`` also
makes the single-model intervals and the paired model-vs-model intervals mutually consistent, and
turns the whole computation into a few BLAS matrix products per chunk.

``W`` is drawn by a **circular moving-block** scheme with block length ``block_days`` (default
:data:`BLOCK_DAYS` = 7, one week — longer than the synoptic decorrelation time of a daily 2 m
temperature error), because consecutive daily forecast errors are not independent: an iid day
bootstrap understates the uncertainty of a window-mean MAE.  ``block_days=1`` recovers the plain
multinomial resample; on the real archive the block version is 20–30 % wider.

*Aggregate ("ALL") rows* — for ``station_id="ALL"`` the daily series is first averaged across the
stations that have a value on that day (separately for |error|, error, error² and each hit
indicator), and the statistics and bootstrap are then computed over days exactly as for a station.
Pooling all station-days instead would weight a day with 23 stations 23× a day with one, and would
break the exchangeability of the day as the resampling unit; ``n_stations`` records the mean number
of stations behind an ALL row so the reader can see how balanced it is (METHODOLOGY §4).

*Model versions* — a model's scores are never aggregated across a cycle/weight change
(METHODOLOGY §7).  Each ``model_id`` is truncated to its most recent contiguous ``model_version``
segment before scoring; ``model_version`` and ``segment_start`` are published with every row.
"""

from __future__ import annotations

import itertools
from datetime import date

import numpy as np
import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION
from .store import DAILY_COLUMNS

__all__ = [
    "BLOCK_DAYS",
    "PAIRWISE_COLUMNS",
    "PERSISTENCE_ID",
    "SCORE_COLUMNS",
    "latest_version_segments",
    "persistence_daily",
    "score",
    "select_truth",
]

PERSISTENCE_ID = "persistence"
F_TO_C = 5.0 / 9.0
HIT_THRESHOLDS_C = (1.0 * F_TO_C, 2.0 * F_TO_C, 3.0 * F_TO_C)
MIN_N = 30  # METHODOLOGY §4: windows below this are published but greyed out
ALL_STATIONS = "ALL"
VARIABLES = ("tmax", "tmin")
#: Moving-block length (days) for the day bootstrap (METHODOLOGY §5).  One week: longer than
#: the synoptic decorrelation time of a daily 2 m temperature error, and the value the public API
#: envelope advertises.
BLOCK_DAYS = 7
UNKNOWN_VERSION = "unknown"

SCORE_COLUMNS = [
    "station_id", "model_id", "init_hour", "lead_day", "variable", "method", "window",
    "n", "n_stations", "n_flagged", "mae", "bias", "rmse", "hit1f", "hit2f", "hit3f",
    "mae_debiased", "skill_persistence", "skill_persistence_debiased",
    "mae_ci_low", "mae_ci_high", "bias_ci_low", "bias_ci_high",
    "rmse_ci_low", "rmse_ci_high", "hit1f_ci_low", "hit1f_ci_high",
    "model_version", "segment_start",
    "period_start", "period_end", "computed_at", "methodology_version", "schema_version",
]

# DESIGN §3.5 columns, plus `method` (the comparison is only meaningful within one extraction
# method; see module docstring of site/build.py for how it is surfaced).
PAIRWISE_COLUMNS = [
    "station_id", "init_hour", "lead_day", "variable", "window", "model_a", "model_b",
    "n_common", "mae_diff", "ci_low", "ci_high", "significant", "method",
    "computed_at", "methodology_version", "schema_version",
]

# METHODOLOGY §3: the truth is the *first final* CLI value.  A same-day preliminary CLI
# ("TODAY ... VALID AS OF") is never used, not even as a last resort — it is dropped, not ranked
# below the hourly-observation fallback.
_TRUTH_RANK = {("CLI", True): 0, ("CF6", False): 1, ("CF6", True): 1, ("OBS", False): 2,
               ("OBS", True): 2}


def empty_scores() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SCORE_COLUMNS})


def empty_pairwise() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PAIRWISE_COLUMNS})


def window_label(w: int | None) -> str:
    return "all" if w is None else f"{int(w)}d"


# --------------------------------------------------------------------------------------------
# truth
# --------------------------------------------------------------------------------------------

def select_truth(truth: pd.DataFrame) -> pd.DataFrame:
    """One observation per ``(station_id, climo_date, variable)`` with the METHODOLOGY §3 priority.

    Returns columns ``station_id, climo_date, variable, obs_c, truth_source, qc_flag``.
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
    for var in VARIABLES:
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
# persistence baseline
# --------------------------------------------------------------------------------------------

def persistence_daily(
    truth: pd.DataFrame,
    leads: tuple[int, ...] = tuple(range(1, 10)),
    methods: tuple[str, ...] = ("bilinear", "nearest"),
    init_hours: tuple[int, ...] = (0, 12),
) -> pd.DataFrame:
    """Persistence baseline in the ``daily_forecasts`` schema (METHODOLOGY §4).

    **Definition (lagged persistence).** For target climatological day ``D`` and lead day ``L`` the
    persistence forecast is the observation of day ``D − L`` — the last observation a forecaster
    issuing at lead ``L`` could already have seen.  ``L = 1`` is classic persistence ("yesterday's
    value"); larger ``L`` degrade it the way a model's skill degrades, so that
    ``skill = 1 − MAE_model/MAE_persistence`` compares like with like at every lead.  Lead day 0 is
    not produced because ``D − 0`` is the target day itself.

    The rejected alternative — "yesterday's observation at every lead" — gives the baseline
    information the forecast did not have (at lead 7 it would be 6 days fresher than the model's
    initial condition), which makes it artificially hard to beat at long leads and hides the whole
    point of the skill curve.  Lagged persistence is the standard denominator for exactly this
    reason.

    **Caveat that the caller must not forget.** The baseline is an *observed* daily extreme, while
    the model columns are 6-hourly *sampled* extremes (METHODOLOGY §2.3).  The sampling bias
    (typically −2 to −4 °C on Tmax) is in the numerator only, so ``skill_persistence`` is negative
    for every model at every lead and is not a statement about forecast quality.  Use
    ``skill_persistence_debiased``, which removes each series' own mean error over the common days,
    to compare the random part of the error.

    Rows are emitted for both initialization hours and both extraction methods (the baseline does
    not depend on either) so that it lines up with every model group in :func:`score`.
    """
    sel = select_truth(truth)
    if sel.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in DAILY_COLUMNS})
    wide = sel.pivot_table(
        index=["station_id", "climo_date"], columns="variable", values="obs_c", aggfunc="first"
    ).reset_index()
    for var in VARIABLES:
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
        return pd.DataFrame({c: pd.Series(dtype="object") for c in DAILY_COLUMNS})
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
    df["tmax_native_c"] = np.float32(np.nan)
    df["tmin_native_c"] = np.float32(np.nan)
    df["missing_reason"] = ""
    df["lead_day"] = df["lead_day"].astype("int8")
    df["climo_date"] = df["climo_date"].dt.date
    df["schema_version"] = SCHEMA_VERSION
    df["methodology_version"] = METHODOLOGY_VERSION
    return df[DAILY_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# error table
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


def _daily_long(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["init_time"] = pd.to_datetime(d["init_time"], utc=True)
    d["climo_date"] = pd.to_datetime(d["climo_date"]).dt.normalize()
    d["init_hour"] = d["init_time"].dt.hour.astype("int16")
    frames = []
    for var in VARIABLES:
        col = f"{var}_sampled_c"
        part = d[["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date"]].copy()
        part["variable"] = var
        part["fcst_c"] = pd.to_numeric(d[col], errors="coerce")
        frames.append(part[part["fcst_c"].notna()])
    if not frames:
        return pd.DataFrame(
            columns=["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date",
                     "variable", "fcst_c"]
        )
    return pd.concat(frames, ignore_index=True)


def error_table(daily: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts to truth and return one row per scored (group, day) with the signed error."""
    long = _daily_long(daily)
    sel = select_truth(truth)
    if long.empty or sel.empty:
        return pd.DataFrame(
            columns=["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date",
                     "variable", "fcst_c", "obs_c", "err"]
        )
    m = long.merge(sel, on=["station_id", "climo_date", "variable"], how="inner")
    m["err"] = m["fcst_c"] - m["obs_c"]
    m = m[m["err"].notna()]
    # DESIGN §3.3 key already guarantees uniqueness; be defensive against duplicated shards
    m = m.drop_duplicates(
        subset=["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date", "variable"]
    )
    return m.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# the scoring machinery
# --------------------------------------------------------------------------------------------

_KEY_COLS = ["station_id", "model_id", "init_hour", "lead_day", "variable", "method"]


#: Per (group, day) functionals carried through the dense matrices.  ``ns`` is the number of
#: stations behind the unit (1 for a station row, the count for an ALL row) and ``fl`` marks a day
#: whose truth carries a ``qc_flag`` (METHODOLOGY §6).
_UNIT_COLS = ("a", "s", "q", "h1", "h2", "h3", "ns", "fl")


def _unit_frame(err: pd.DataFrame) -> pd.DataFrame:
    """Per (group, day) error functionals, including the cross-station ``ALL`` rows."""
    e = err
    a = e["err"].abs().to_numpy(dtype="float64")
    s = e["err"].to_numpy(dtype="float64")
    unit = e[_KEY_COLS + ["climo_date"]].copy()
    unit["a"] = a
    unit["s"] = s
    unit["q"] = s * s
    for i, thr in enumerate(HIT_THRESHOLDS_C, start=1):
        # METHODOLOGY §4: the threshold is inclusive; |err| == 1 °F counts as a hit.  The epsilon
        # absorbs the float32 → float64 round-trip of a value that is exactly 1 °F in °C.
        unit[f"h{i}"] = (a <= thr + 1e-9).astype("float64")
    unit["ns"] = 1.0
    flag = e["qc_flag"].astype(str) if "qc_flag" in e else pd.Series("", index=e.index)
    unit["fl"] = (flag.fillna("").to_numpy() != "").astype("float64")

    grp = ["model_id", "init_hour", "lead_day", "variable", "method", "climo_date"]
    allrows = (
        unit.groupby(grp, observed=True)
        .agg(
            a=("a", "mean"), s=("s", "mean"), q=("q", "mean"),
            h1=("h1", "mean"), h2=("h2", "mean"), h3=("h3", "mean"),
            ns=("ns", "sum"),
            fl=("fl", "max"),  # a day is flagged for ALL as soon as any of its stations is
        )
        .reset_index()
    )
    allrows["station_id"] = ALL_STATIONS
    return pd.concat(
        [unit, allrows[_KEY_COLS + ["climo_date"] + list(_UNIT_COLS)]], ignore_index=True
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


def _resample_weights(n_days: int, n_boot: int, block_days: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_boot, n_days)`` day-resample counts, shared by every group in the window.

    ``block_days <= 1`` draws a plain multinomial (iid day) resample.  Otherwise a *circular moving
    block* bootstrap is used: ``ceil(n_days / L)`` blocks of ``L`` consecutive days are drawn with
    replacement from a uniformly random start and wrapped around the window, and the first
    ``n_days`` of the concatenation are counted.  Blocks keep the day-to-day autocorrelation of the
    error series inside the resample, which an iid resample destroys (METHODOLOGY §5).

    The block length is clipped to ``max(1, n_days // 4)`` so that a short window still has at least
    four blocks to draw from.
    """
    if block_days <= 1 or n_days < 4:
        return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot).astype("float32")
    L = int(min(block_days, max(1, n_days // 4)))
    if L <= 1:
        return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot).astype("float32")
    n_blocks = int(np.ceil(n_days / L))
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n_days
    idx = idx.reshape(n_boot, -1)[:, :n_days]
    flat = idx + (np.arange(n_boot, dtype=idx.dtype) * n_days)[:, None]
    W = np.bincount(flat.ravel(), minlength=n_boot * n_days).astype("float32")
    return W.reshape(n_boot, n_days)


def _percentiles(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5/97.5 percentiles down the bootstrap axis (NaN-safe, but fast when there is no NaN)."""
    with np.errstate(invalid="ignore"):
        fn = np.nanpercentile if np.isnan(boot).any() else np.percentile
        lo, hi = fn(boot, [2.5, 97.5], axis=0)
    return lo, hi


def score(
    daily: pd.DataFrame,
    truth: pd.DataFrame,
    windows: tuple[int | None, ...] = (30, 90, 365, None),
    n_boot: int = 1000,
    seed: int = 0,
    as_of: date | str | None = None,
    pairwise_methods: tuple[str, ...] | None = None,
    max_cells_per_chunk: int = 400,
    block_days: int = BLOCK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the published ``scores`` and ``pairwise`` tables.

    ``pairwise_methods`` restricts the (expensive) model-vs-model comparison to a subset of
    extraction methods; ``None`` means all of them.  Set ``n_boot=0`` to skip the bootstrap.
    ``block_days`` is the moving-block length of the day bootstrap (METHODOLOGY §5); ``1`` gives the
    plain iid day resample.
    """
    computed_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    if daily is None or len(daily) == 0 or truth is None or len(truth) == 0:
        return empty_scores(), empty_pairwise()

    daily = daily.copy()
    if PERSISTENCE_ID not in set(daily["model_id"].astype(str)):
        model_leads = pd.to_numeric(daily["lead_day"], errors="coerce").dropna().astype(int)
        leads = tuple(sorted({int(x) for x in model_leads.unique() if x >= 1})) or (1,)
        pers = persistence_daily(truth, leads=leads)
        if len(pers):
            daily = pd.concat([daily, pers], ignore_index=True)

    # METHODOLOGY §7: never aggregate across a cycle / weight change
    segments = latest_version_segments(daily)
    if len(segments):
        daily = daily.merge(segments, on="model_id", how="left", suffixes=("", "_seg"))
        keep = pd.to_datetime(daily["init_time"], utc=True) >= pd.to_datetime(
            daily["segment_start"], utc=True
        )
        daily = daily[keep.fillna(True)].reset_index(drop=True)

    err = error_table(daily, truth)
    if err.empty:
        return empty_scores(), empty_pairwise()

    if as_of is None:
        as_of_ts = pd.to_datetime(truth["climo_date"]).max().normalize()
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

    # window → (first date index, weight matrix)
    rng = np.random.default_rng(seed)
    win_defs: list[tuple[str, int, np.ndarray | None]] = []
    for w in windows:
        if w is None:
            d0 = 0
        else:
            cutoff = as_of_ts - pd.Timedelta(days=int(w) - 1)
            d0 = int(np.searchsorted(dates_ts.values, np.datetime64(cutoff), side="left"))
        dw = n_dates - d0
        if dw <= 0:
            continue
        W = _resample_weights(dw, n_boot, block_days, rng) if (n_boot and dw >= 2) else None
        win_defs.append((window_label(w), d0, W))

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

        for label, d0, W in win_defs:
            Mw = M[d0:]
            n = Mw.sum(axis=0)
            live = n > 0
            if not live.any():
                continue
            Aw, Sw, Qw = dense["a"][d0:], dense["s"][d0:], dense["q"][d0:]
            H1w = dense["h1"][d0:]
            is_pers = model_of_col[lo_col:hi_col] == PERSISTENCE_ID
            with np.errstate(invalid="ignore", divide="ignore"):
                mae = Aw.sum(axis=0) / n
                bias = Sw.sum(axis=0) / n
                rmse = np.sqrt(Qw.sum(axis=0) / n)
                hits = [dense[f"h{i}"][d0:].sum(axis=0) / n for i in (1, 2, 3)]
                n_stations = dense["ns"][d0:].sum(axis=0) / n
                n_flagged = dense["fl"][d0:].sum(axis=0)
                # MAE after removing each series' own mean error over the window: separates the
                # random part of the error from the constant sampling/representativeness offset
                mae_db = (np.abs(Sw - bias[None, :]) * Mw).sum(axis=0) / n

                common = Mw * Mw[:, p_local_idx]
                nc = common.sum(axis=0)
                mae_self = (Aw * common).sum(axis=0) / nc
                mae_pers = (Aw[:, p_local_idx] * common).sum(axis=0) / nc
                skill = 1.0 - mae_self / mae_pers
                # the debiased twin, with both series de-meaned over the *common* days
                Sp = Sw[:, p_local_idx]
                b_self = (Sw * common).sum(axis=0) / nc
                b_pers = (Sp * common).sum(axis=0) / nc
                db_self = (np.abs(Sw - b_self[None, :]) * common).sum(axis=0) / nc
                db_pers = (np.abs(Sp - b_pers[None, :]) * common).sum(axis=0) / nc
                skill_db = 1.0 - db_self / db_pers
            ok_pers = has_pers & (nc > 0) & ~is_pers
            skill = np.where(ok_pers & (mae_pers > 0), skill, np.nan)
            skill_db = np.where(ok_pers & (db_pers > 0), skill_db, np.nan)

            if W is not None:
                stacked = np.concatenate([Aw, Sw, Qw, H1w, Mw], axis=1)
                bmat = W @ stacked
                den = bmat[:, 4 * gc:]
                den[den == 0] = np.nan
                mae_lo, mae_hi = _percentiles(bmat[:, :gc] / den)
                bias_lo, bias_hi = _percentiles(bmat[:, gc: 2 * gc] / den)
                rmse_lo, rmse_hi = _percentiles(np.sqrt(bmat[:, 2 * gc: 3 * gc] / den))
                hit1_lo, hit1_hi = _percentiles(bmat[:, 3 * gc: 4 * gc] / den)
                del bmat, stacked
            else:
                nanv = np.full(gc, np.nan, dtype="float32")
                mae_lo = mae_hi = bias_lo = bias_hi = nanv
                rmse_lo = rmse_hi = hit1_lo = hit1_hi = nanv

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
                "mae_debiased": mae_db,
                "skill_persistence": skill,
                "skill_persistence_debiased": skill_db,
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
                        label=label, W=W, Aw=Aw, Mw=Mw, lo_col=lo_col, hi_col=hi_col,
                        cell_of_col=cell_of_col, model_of_col=model_of_col, col_df=col_df,
                        ok_cells=pw_ok_cell,
                    )
                )
        del dense, M

    scores = (
        pd.concat(score_parts, ignore_index=True) if score_parts else empty_scores()
    )
    if len(scores):
        if len(segments):
            seg = segments.copy()
            seg["segment_start"] = pd.to_datetime(seg["segment_start"], utc=True).dt.date
            scores = scores.merge(seg, on="model_id", how="left")
        else:
            scores["model_version"] = UNKNOWN_VERSION
            scores["segment_start"] = pd.NaT
        scores["computed_at"] = computed_at
        scores["methodology_version"] = METHODOLOGY_VERSION
        scores["schema_version"] = SCHEMA_VERSION
        scores = scores[SCORE_COLUMNS]
        scores = scores.sort_values(
            ["station_id", "variable", "window", "lead_day", "init_hour", "method", "mae"]
        ).reset_index(drop=True)

    pairwise = pd.concat(pair_parts, ignore_index=True) if pair_parts else empty_pairwise()
    if len(pairwise):
        pairwise["computed_at"] = computed_at
        pairwise["methodology_version"] = METHODOLOGY_VERSION
        pairwise["schema_version"] = SCHEMA_VERSION
        pairwise = pairwise[PAIRWISE_COLUMNS].reset_index(drop=True)
    return scores, pairwise


def _pairwise_chunk(
    *, label, W, Aw, Mw, lo_col, hi_col, cell_of_col, model_of_col, col_df, ok_cells,
    max_pair_cols: int = 6000,
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
        common = Mw[:, sa] * Mw[:, sb]
        nc = common.sum(axis=0)
        keep = nc > 0
        if not keep.any():
            continue
        dnum = (Aw[:, sa] - Aw[:, sb]) * common
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = dnum.sum(axis=0) / nc
        if W is not None:
            bmat = W @ np.concatenate([dnum, common], axis=1)
            k = dnum.shape[1]
            den = bmat[:, k:]
            den[den == 0] = np.nan
            lo, hi = _percentiles(bmat[:, :k] / den)
            del bmat
        else:
            lo = hi = np.full(len(sa), np.nan, dtype="float32")
        sig = np.isfinite(lo) & np.isfinite(hi) & ((lo > 0) | (hi < 0))
        out.append(pd.DataFrame({
            "station_id": station[sa], "init_hour": init_hour[sa], "lead_day": lead[sa],
            "variable": variable[sa], "window": label, "model_a": models[sa],
            "model_b": models[sb], "n_common": nc.astype("int32"), "mae_diff": diff,
            "ci_low": lo, "ci_high": hi, "significant": sig, "method": method[sa],
        })[keep].reset_index(drop=True))
    return out
