"""Parquet storage of monthly/yearly shards with idempotent upserts (DESIGN §3)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DATA_DIR
from .sources.base import FORECAST_VALUE_COLUMNS, FORECAST_VALUE_KEY

log = logging.getLogger(__name__)

TRUTH_COLUMNS = [
    "station_id", "climo_date", "source", "tmax_f", "tmin_f", "tmax_c", "tmin_c", "issuance_time",
    "is_final", "revised", "revised_tmax_f", "revised_tmin_f", "qc_flag", "product_id",
    "schema_version", "methodology_version",
]
TRUTH_KEY = ["station_id", "climo_date", "source"]

#: Observed 2 m temperature at the four synoptic instants (DESIGN §10.1). Built by
#: :mod:`castcheck.truth_instant`; ``source`` is ``ASOS_IEM`` (archive) or ``NWS_API`` (last week).
TRUTH_INSTANT_COLUMNS = [
    "station_id", "valid_time", "temp_c", "obs_time", "source", "n_reports", "qc_flag",
    "schema_version", "methodology_version",
]
TRUTH_INSTANT_KEY = ["station_id", "valid_time"]

DAILY_COLUMNS = [
    "model_id", "model_version", "init_time", "station_id", "climo_date", "lead_day", "method",
    "tmax_sampled_c", "tmin_sampled_c", "n_samples", "tmax_native_c", "tmin_native_c",
    "missing_reason",
    # v0.3 (DESIGN §10.2): the observed sampled extremes that make `tmax_s`/`tmin_s` like-for-like,
    # and how far past the climatological day a native-extreme accumulation window reaches.
    # `castcheck.derive.daily_columns()` produces them in exactly this position.
    "tmax_obs_s_c", "tmin_obs_s_c", "n_obs_samples", "native_overhang_h",
    "schema_version", "methodology_version",
]
DAILY_KEY = ["model_id", "init_time", "station_id", "climo_date", "method"]


_FV_DTYPES = {"lead_h": "int16", "bucket_h": "int8", "value_c": "float32"}
_TRUTH_DTYPES = {"tmax_f": "Int16", "tmin_f": "Int16", "revised_tmax_f": "Int16", "revised_tmin_f": "Int16",
                 "tmax_c": "float32", "tmin_c": "float32", "is_final": "bool", "revised": "bool"}
# The v0.3 integer columns are *nullable*: rows derived before they existed, and models with no
# native extremes at all, have no value for them, and plain int8 has no way to say so.
_DAILY_DTYPES = {"lead_day": "int8", "n_samples": "int8", "tmax_sampled_c": "float32", "tmin_sampled_c": "float32",
                 "tmax_native_c": "float32", "tmin_native_c": "float32",
                 "tmax_obs_s_c": "float32", "tmin_obs_s_c": "float32",
                 "n_obs_samples": "Int8", "native_overhang_h": "Int8"}
_TRUTH_INSTANT_DTYPES = {"temp_c": "float32", "n_reports": "int8"}


def _cast(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Narrow column dtypes per DESIGN §3.

    Tolerant, but never *silently* tolerant: a column that cannot be narrowed is logged, because a
    column that keeps drifting back to ``float64`` changes what the parquet file says on disk and
    what the next read gets back.
    """
    for col, dt in dtypes.items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].astype(dt)
            continue
        except (TypeError, ValueError):
            pass
        if dt.startswith(("Int", "UInt")) and pd.api.types.is_numeric_dtype(df[col]):
            # float column carrying NaN: pandas refuses float64 -> Int16 directly, so go via the
            # nullable float dtype, where NaN becomes pd.NA. Restricted to columns that are already
            # numeric — coercing text to NA here would silently destroy data instead of reporting it.
            try:
                df[col] = df[col].round().astype("Float64").astype(dt)
                continue
            except (TypeError, ValueError):
                pass
        log.warning("could not cast column %r to %s (left as %s)", col, dt, df[col].dtype)
    return df


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_shard(path: Path) -> bool:
    """Whether a file matched by a shard glob is a real shard.

    :func:`_write` stages every shard next to its destination so that the rename is atomic on the
    same filesystem. A staging file that a crash or a concurrent writer leaves behind must not be
    picked up as data: a half-written parquet makes the *reader* fail with ``ArrowInvalid``, which
    is a failure in a completely unrelated command. Dotfiles are skipped for the same reason (and
    because macOS drops ``._`` resource forks into synced directories).

    Names containing a space are skipped too. No shard this project writes has one — every name is
    ``key=value.parquet`` — so a space means a file-sync conflict copy (``year_month=2026-08
    2.parquet``), which is a *byte-identical duplicate* of a real shard and would silently double
    every row it holds when the directory is read.
    """
    name = path.name
    return not name.startswith(".") and ".tmp" not in name and " " not in name


def _shards(base: Path, pattern: str) -> list[Path]:
    """Shard files under `base` matching `pattern`, staging files and dotfiles excluded."""
    if not base.exists():
        return []
    return sorted(p for p in base.glob(pattern) if _is_shard(p))


def _conform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Give `df` exactly `columns`, adding any the file predates as all-missing.

    Schema growth is normal here (``native_overhang_h`` arrived in v0.3), and a table written before
    a column existed must still read back with that column rather than raising a ``KeyError`` three
    call frames away in whatever wanted it.
    """
    missing = [c for c in columns if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    if missing:
        log.debug("added %d absent column(s) to a stored shard: %s", len(missing), missing)
    return df[columns]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pq.read_table(path).to_pandas()
    return _conform(df, columns) if columns else df


def _set_where(target: pd.DataFrame, take: np.ndarray, updates: pd.DataFrame, cols: list[str]) -> None:
    """``target.loc[take, cols] = updates[cols]`` done dtype-safely, in place.

    Assigning an *index-aligned* DataFrame into a pandas nullable-integer column via ``.loc`` hits a
    pandas bug (``KeyError: np.int64(N)``): the lossless-cast check inside
    ``pandas.core.arrays.numeric._coerce_to_data_and_mask`` indexes the value **Series** positionally
    while that Series carries a (Multi)Index of labels. It only fires when the column holds a mix of
    present and missing values, which is exactly the shape of ``revised_tmax_f`` over a long
    backfill. Going through ``object`` and restoring the dtype afterwards avoids the code path
    entirely and keeps the column's declared dtype.
    """
    for col in cols:
        dtype = target[col].dtype
        aligned = updates[col].reindex(target.index)
        merged = target[col].astype(object).where(~take, aligned.astype(object))
        target[col] = merged.astype(dtype)


#: One lock per shard path. ``backfill --workers N`` fans its fetches out as *threads* in a single
#: process, so two of them routinely land on the same monthly shard. Serialising the whole
#: read-merge-write cycle is what makes the upsert correct: without it the second writer merges into
#: a snapshot taken before the first one landed and silently drops the first writer's rows.
_SHARD_LOCKS: dict[Path, threading.Lock] = {}
_SHARD_LOCKS_GUARD = threading.Lock()


def _shard_lock(path: Path) -> threading.Lock:
    with _SHARD_LOCKS_GUARD:
        return _SHARD_LOCKS.setdefault(path, threading.Lock())


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` atomically.

    The staging name is a dotfile that does **not** end in ``.parquet``, so no shard glob can match
    it while it is being written; :func:`_is_shard` skips it even if a crash leaves it behind. It is
    also unique per writer: the pid alone is not, because the backfill's workers are threads in one
    process, and two of them staging into the same name means one renames a half-written file into
    place (observed as ``Invalid data`` / ``FileNotFoundError`` on the next read).
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=_ensure(path).parent)
    os.close(fd)
    tmp = Path(name)
    try:
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _upsert(existing: pd.DataFrame, new: pd.DataFrame, key: list[str], prefer_present_col: str | None) -> pd.DataFrame:
    """Concatenate and keep one row per key: present values beat missing ones, later fetch beats earlier."""
    df = pd.concat([existing, new], ignore_index=True)
    if df.empty:
        return df
    if prefer_present_col is not None:
        df["_present"] = (df[prefer_present_col] == "").astype(int)
        sort_cols = key + ["_present", "fetched_at"]
    else:
        df["_present"] = 1
        sort_cols = key + ["_present"]
    df = df.sort_values(sort_cols).drop_duplicates(subset=key, keep="last").drop(columns="_present")
    return df.reset_index(drop=True)


# ---------------- forecast_values ----------------

def forecast_values_path(model_id: str, year_month: str) -> Path:
    return DATA_DIR / "forecast_values" / f"model_id={model_id}" / f"year_month={year_month}.parquet"


def upsert_forecast_values(df: pd.DataFrame) -> dict[str, int]:
    """Upsert rows into monthly shards keyed by init month. Returns {shard: n_rows}."""
    if df.empty:
        return {}
    df = df[FORECAST_VALUE_COLUMNS].copy()
    df["init_time"] = pd.to_datetime(df["init_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    out: dict[str, int] = {}
    ym = df["init_time"].dt.strftime("%Y-%m")
    for (model_id, year_month), part in df.groupby([df["model_id"], ym]):
        path = forecast_values_path(model_id, year_month)
        with _shard_lock(path):
            merged = _upsert(_read(path, FORECAST_VALUE_COLUMNS), part, FORECAST_VALUE_KEY, "missing_reason")
            _write(_cast(merged, _FV_DTYPES), path)
        out[str(path.relative_to(DATA_DIR))] = len(merged)
    return out


def forecast_value_shards(model_ids: list[str] | None = None, start: str | None = None,
                          end: str | None = None) -> list[Path]:
    """Monthly shard files matching a model subset and an inclusive ``YYYY-MM-DD`` init-date range.

    Shards are named by init month, so restricting the range restricts the files that are opened at
    all — this is what keeps :func:`existing_inits` O(window) instead of O(archive).
    """
    base = DATA_DIR / "forecast_values"
    if not base.exists():
        return []
    out: list[Path] = []
    for mdir in sorted(p for p in base.glob("model_id=*") if p.is_dir()):
        mid = mdir.name.split("=", 1)[1]
        if model_ids and mid not in model_ids:
            continue
        for f in _shards(mdir, "year_month=*.parquet"):
            ym = f.stem.split("=", 1)[1]
            if start and ym < start[:7]:
                continue
            if end and ym > end[:7]:
                continue
            out.append(f)
    return out


def read_forecast_values(model_ids: list[str] | None = None, start: str | None = None,
                         end: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Read `forecast_values` shards.

    `columns` restricts the parquet column projection (the file format is columnar, so reading four
    of fifteen columns costs roughly a quarter of the I/O and a fraction of the memory). `init_time`
    is always read when a date range is given, because the range is applied row-wise as well.
    """
    want = list(columns) if columns else None
    if want is not None and (start or end) and "init_time" not in want:
        want = [*want, "init_time"]
    frames = [pq.read_table(f, columns=want).to_pandas()
              for f in forecast_value_shards(model_ids, start, end)]
    if not frames:
        return pd.DataFrame(columns=want or FORECAST_VALUE_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if start:
        df = df[df["init_time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["init_time"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


#: columns `existing_inits` needs; everything else stays on disk.
_COMPLETENESS_COLUMNS = ["init_time", "valid_time", "variable", "missing_reason", "lead_h"]


def existing_inits(model_id: str, min_valid: int | None = None, start: str | None = None,
                   end: str | None = None) -> set[pd.Timestamp]:
    """Init times whose run is *complete enough* to skip: at least `min_valid` distinct **forecast**
    valid times with a present t2 value (default: max_h/step_h from models.yaml, i.e. every 6-hourly
    step out to 240 h). Partial runs (e.g. a 503 on one step) are therefore retried by later
    fetch-latest/backfill passes.

    The analysis step ``lead_h == 0`` is excluded from the count. AIWP stores f000 and the GRIB
    sources do not, so counting every stored step made a 41-row AIWP run that was missing f240 tie
    the 40-step threshold and be declared complete for ever — the one run shape this function exists
    to catch. Counting forecast steps only makes the threshold mean the same thing for every source.

    `start`/`end` (inclusive ``YYYY-MM-DD``) restrict the scan to the months a caller actually cares
    about; callers that plan work over a bounded window (``fetch-latest``, ``backfill``) must pass
    them, otherwise the whole archive is read on every invocation.
    """
    from .config import model_by_id

    df = read_forecast_values([model_id], start=start, end=end, columns=_COMPLETENESS_COLUMNS)
    if df.empty:
        return set()
    step_h = 6
    if min_valid is None:
        try:
            m = model_by_id(model_id)
            step_h = max(int(m.step_h), 1)
            min_valid = m.max_h // step_h
        except KeyError:
            min_valid = 1
    ok = df[(df["variable"] == "t2") & (df["missing_reason"] == "")]
    if "lead_h" in ok.columns:
        ok = ok[pd.to_numeric(ok["lead_h"], errors="coerce").fillna(step_h) >= step_h]
    if ok.empty:
        return set()
    n_valid = ok.groupby("init_time")["valid_time"].nunique()
    return set(n_valid[n_valid >= min_valid].index)


def last_attempt_by_init(model_id: str, start: str | None = None,
                         end: str | None = None) -> dict[pd.Timestamp, pd.Timestamp]:
    """``init_time -> newest fetched_at`` for the stored shards, i.e. when we last tried each run.

    `fetch-latest` uses this to avoid re-downloading a run that upstream never completed on every
    scheduled pass. Two columns only, and only the shards covering ``[start, end]``.
    """
    df = read_forecast_values([model_id], start=start, end=end, columns=["init_time", "fetched_at"])
    if df.empty:
        return {}
    newest = df.groupby("init_time")["fetched_at"].max()
    return {k: pd.Timestamp(v) for k, v in newest.items()}


# ---------------- truth_daily ----------------

def truth_path(year: int) -> Path:
    return DATA_DIR / "truth_daily" / f"year={year}.parquet"


def _apply_first_final(merged: pd.DataFrame) -> pd.DataFrame:
    """First-final policy (METHODOLOGY §3).

    The earliest issuance per ``(station_id, climo_date, source)`` supplies the published values.
    Later issuances contribute only their correction fields (``revised``, ``revised_tmax_f``,
    ``revised_tmin_f``) and the union of the quality flags, so a correction or a QC flag discovered
    on a later run is recorded without ever changing a published value.
    """
    merged = merged.sort_values(TRUTH_KEY + ["issuance_time"], kind="stable")
    first = merged.drop_duplicates(subset=TRUTH_KEY, keep="first").set_index(TRUTH_KEY)

    revisions = merged[merged["revised"].fillna(False).astype(bool)]
    if len(revisions):
        latest = revisions.drop_duplicates(subset=TRUTH_KEY, keep="last").set_index(TRUTH_KEY)
        take = first.index.isin(latest.index)
        if take.any():
            _set_where(first, take, latest, ["revised", "revised_tmax_f", "revised_tmin_f"])

    flags = merged.groupby(TRUTH_KEY, sort=False)["qc_flag"].agg(
        lambda s: ";".join(dict.fromkeys(p for v in s for p in str(v or "").split(";") if p))
    )
    first["qc_flag"] = flags.reindex(first.index).fillna("")
    return first.reset_index()[TRUTH_COLUMNS]


def upsert_truth(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    df = df[TRUTH_COLUMNS].copy()
    df["climo_date"] = pd.to_datetime(df["climo_date"]).dt.date
    df["issuance_time"] = pd.to_datetime(df["issuance_time"], utc=True)
    out = {}
    for year, part in df.groupby(pd.to_datetime(df["climo_date"]).dt.year):
        path = truth_path(int(year))
        existing = _read(path, TRUTH_COLUMNS)
        if existing.empty:
            merged = part.copy()
        else:
            existing["climo_date"] = pd.to_datetime(existing["climo_date"]).dt.date
            existing["issuance_time"] = pd.to_datetime(existing["issuance_time"], utc=True)
            merged = pd.concat([existing, part], ignore_index=True)
        merged = _apply_first_final(merged)
        merged = merged.sort_values(TRUTH_KEY).reset_index(drop=True)
        _write(_cast(merged, _TRUTH_DTYPES), path)
        out[str(path.relative_to(DATA_DIR))] = len(merged)
    return out


def overwrite_truth(df: pd.DataFrame) -> dict[str, int]:
    """Replace whole ``truth_daily`` year shards with `df`, bypassing the first-final merge.

    :func:`upsert_truth` exists to make *arrival order* irrelevant: whatever it is handed, the
    earliest issuance wins. That is exactly wrong for re-running a quality check over rows that are
    already stored — the stored row and the corrected row carry the same key *and the same issuance
    time*, so the merge would keep the stored one and the correction would vanish without a trace.

    This is therefore the only writer that overwrites. The caller must pass **every row of each year
    it touches** (``castcheck truth-qc`` reads whole years for this reason); a partial frame would
    delete the rest of the year.
    """
    if df.empty:
        return {}
    df = df[TRUTH_COLUMNS].copy()
    df["climo_date"] = pd.to_datetime(df["climo_date"]).dt.date
    df["issuance_time"] = pd.to_datetime(df["issuance_time"], utc=True)
    out = {}
    for year, part in df.groupby(pd.to_datetime(df["climo_date"]).dt.year):
        path = truth_path(int(year))
        merged = part.sort_values(TRUTH_KEY).reset_index(drop=True)
        _write(_cast(merged, _TRUTH_DTYPES), path)
        out[str(path.relative_to(DATA_DIR))] = len(merged)
    return out


def read_truth(years: list[int] | None = None) -> pd.DataFrame:
    frames = [_read(f, TRUTH_COLUMNS) for f in _shards(DATA_DIR / "truth_daily", "year=*.parquet")
              if not years or int(f.stem.split("=")[1]) in years]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRUTH_COLUMNS)


# ---------------- truth_instant (DESIGN §10.1) ----------------

def truth_instant_path(year: int) -> Path:
    return DATA_DIR / "truth_instant" / f"year={year}.parquet"


#: source ranking for :func:`resolve_truth_instant` — higher wins among rows that both carry a value
_INSTANT_SOURCE_RANK = {"NWS_API": 0, "ASOS_IEM": 1}


def resolve_truth_instant(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(station_id, valid_time)``: a value beats a gap, the archive beats the API.

    Shared by :func:`upsert_truth_instant` and :func:`castcheck.merge.merge_frames` so that a shard
    resolved on disk and a shard resolved during a git conflict cannot disagree.
    """
    if df.empty:
        return df
    df = df.copy()
    df["_has_value"] = df["temp_c"].notna().astype(int)
    df["_src"] = df["source"].map(lambda s: _INSTANT_SOURCE_RANK.get(str(s), -1))
    df = (df.sort_values(TRUTH_INSTANT_KEY + ["_has_value", "_src"], kind="stable")
          .drop_duplicates(subset=TRUTH_INSTANT_KEY, keep="last")
          .drop(columns=["_has_value", "_src"]))
    return df.sort_values(TRUTH_INSTANT_KEY).reset_index(drop=True)[TRUTH_INSTANT_COLUMNS]


def upsert_truth_instant(df: pd.DataFrame) -> dict[str, int]:
    """Upsert ``truth_instant`` rows into yearly shards, keyed by ``(station_id, valid_time)``.

    Two sources write the same instants: api.weather.gov covers the last week so that the daily
    pipeline can score yesterday, and the IEM ASOS archive arrives later and is authoritative (it is
    the raw METAR stream rather than a re-served subset, and it does not expire). The resolution
    order is therefore *a value beats a gap, and among values the archive beats the API* — never the
    other way round, because letting an archive row with no observation replace a good API value
    would turn a recovered day back into a hole.
    """
    if df.empty:
        return {}
    df = df[TRUTH_INSTANT_COLUMNS].copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)
    out: dict[str, int] = {}
    for year, part in df.groupby(df["valid_time"].dt.year):
        path = truth_instant_path(int(year))
        existing = _read(path, TRUTH_INSTANT_COLUMNS)
        if not existing.empty:
            existing["valid_time"] = pd.to_datetime(existing["valid_time"], utc=True)
            existing["obs_time"] = pd.to_datetime(existing["obs_time"], utc=True)
        merged = resolve_truth_instant(pd.concat([existing, part], ignore_index=True))
        _write(_cast(merged, _TRUTH_INSTANT_DTYPES), path)
        out[str(path.relative_to(DATA_DIR))] = len(merged)
    return out


def read_truth_instant(years: list[int] | None = None) -> pd.DataFrame:
    frames = [_read(f, TRUTH_INSTANT_COLUMNS)
              for f in _shards(DATA_DIR / "truth_instant", "year=*.parquet")
              if not years or int(f.stem.split("=")[1]) in years]
    if not frames:
        return pd.DataFrame(columns=TRUTH_INSTANT_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)
    return df


# ---------------- daily_forecasts ----------------

def daily_path(model_id: str, year: int) -> Path:
    return DATA_DIR / "daily_forecasts" / f"model_id={model_id}" / f"year={year}.parquet"


def write_daily(df: pd.DataFrame) -> dict[str, int]:
    """Upsert derived daily rows into the ``(model_id, year)`` shard, keyed by :data:`DAILY_KEY`.

    This used to overwrite each shard outright, which is correct only while ``derive`` always
    re-derives the whole archive. Once ``derive`` takes a date window — and it has to, because
    re-deriving three years of values on every daily run is minutes of work to change one day —
    an overwrite would delete every row of that year outside the window. Merging on the key keeps
    a windowed re-derivation idempotent (a re-derived row replaces its older self) without it
    being destructive.
    """
    if df.empty:
        return {}
    df = _conform(df.copy(), DAILY_COLUMNS)
    out = {}
    years = pd.to_datetime(df["climo_date"]).dt.year
    for (model_id, year), part in df.groupby([df["model_id"], years]):
        path = daily_path(model_id, int(year))
        existing = _read(path, DAILY_COLUMNS)
        merged = part.reset_index(drop=True)
        if not existing.empty:
            for frame in (existing, merged):
                frame["climo_date"] = pd.to_datetime(frame["climo_date"]).dt.date
                frame["init_time"] = pd.to_datetime(frame["init_time"], utc=True)
            merged = (pd.concat([existing, merged], ignore_index=True)
                      .drop_duplicates(subset=DAILY_KEY, keep="last"))
        merged = merged.sort_values(DAILY_KEY).reset_index(drop=True)
        _write(_cast(merged, _DAILY_DTYPES), path)
        out[str(path.relative_to(DATA_DIR))] = len(merged)
    return out


def read_daily(model_ids: list[str] | None = None) -> pd.DataFrame:
    base = DATA_DIR / "daily_forecasts"
    if not base.exists():
        return pd.DataFrame(columns=DAILY_COLUMNS)
    frames = []
    for mdir in sorted(p for p in base.glob("model_id=*") if p.is_dir()):
        mid = mdir.name.split("=", 1)[1]
        if model_ids and mid not in model_ids:
            continue
        frames += [_read(f, DAILY_COLUMNS) for f in _shards(mdir, "year=*.parquet")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_COLUMNS)


# ---------------- scores ----------------

def write_scores(scores: pd.DataFrame, pairwise: pd.DataFrame, as_of: str) -> None:
    _write(scores, DATA_DIR / "scores" / "latest.parquet")
    _write(pairwise, DATA_DIR / "scores" / "pairwise_latest.parquet")
    _write(scores, DATA_DIR / "scores" / "history" / f"{as_of}.parquet")


def read_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = DATA_DIR / "scores" / "latest.parquet"
    p = DATA_DIR / "scores" / "pairwise_latest.parquet"
    return (_read(s, []), _read(p, []))


# ---------------- run journal (DESIGN §7.1) ----------------

LAST_RUN_PATH = DATA_DIR / "raw" / "last_run.json"


def read_last_run() -> dict:
    """The run journal, or ``{"commands": {}}`` when nothing has been recorded yet.

    A corrupt file is treated as empty: the journal is observability, never a data dependency.
    """
    try:
        doc = json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"commands": {}}
    if not isinstance(doc, dict) or not isinstance(doc.get("commands"), dict):
        return {"commands": {}}
    return doc


def record_run(command: str, *, status: str, summary: str = "", started_at: datetime | None = None,
               duration_s: float | None = None, exit_code: int = 0, extra: dict | None = None) -> dict:
    """Append one command's outcome to ``data/raw/last_run.json`` (read by ``status.py``).

    The file is a single JSON object ``{"updated_at": iso, "commands": {name: entry}}``; each entry
    carries ``status`` (``"ok"``/``"error"``), ``finished_at``, ``duration_s``, ``exit_code``,
    ``summary`` and a sticky ``last_success_at`` that is only advanced by a successful run, so a
    failing command still shows how stale the last good result is. Writes are atomic; a failure to
    write is logged and swallowed.
    """
    from . import __version__

    now = datetime.now(UTC).replace(microsecond=0)
    doc = read_last_run()
    prev = doc["commands"].get(command, {})
    entry = {
        "status": status,
        "started_at": (started_at or now).replace(microsecond=0).isoformat(),
        "finished_at": now.isoformat(),
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
        "exit_code": int(exit_code),
        # one line, bounded: the journal is a status signal, not a log file
        "summary": " ".join(str(summary).split())[:500],
        "castcheck_version": __version__,
        "last_success_at": now.isoformat() if status == "ok" else prev.get("last_success_at"),
    }
    if extra:
        entry.update(extra)
    doc["commands"][command] = entry
    doc["updated_at"] = now.isoformat()
    try:
        LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=LAST_RUN_PATH.parent,
                                         prefix=".last_run.", suffix=".json", delete=False) as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
            tmp = Path(fh.name)
        os.replace(tmp, LAST_RUN_PATH)
    except OSError as exc:  # observability must never break the pipeline
        log.warning("could not write %s: %s", LAST_RUN_PATH, exc)
    return entry
