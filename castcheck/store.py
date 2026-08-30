"""Parquet storage of monthly/yearly shards with idempotent upserts (DESIGN §3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DATA_DIR
from .sources.base import FORECAST_VALUE_COLUMNS, FORECAST_VALUE_KEY

TRUTH_COLUMNS = [
    "station_id", "climo_date", "source", "tmax_f", "tmin_f", "tmax_c", "tmin_c", "issuance_time",
    "is_final", "revised", "revised_tmax_f", "revised_tmin_f", "qc_flag", "product_id",
    "schema_version", "methodology_version",
]
TRUTH_KEY = ["station_id", "climo_date", "source"]

DAILY_COLUMNS = [
    "model_id", "model_version", "init_time", "station_id", "climo_date", "lead_day", "method",
    "tmax_sampled_c", "tmin_sampled_c", "n_samples", "tmax_native_c", "tmin_native_c", "missing_reason",
    "schema_version", "methodology_version",
]
DAILY_KEY = ["model_id", "init_time", "station_id", "climo_date", "method"]


_FV_DTYPES = {"lead_h": "int16", "bucket_h": "int8", "value_c": "float32"}
_TRUTH_DTYPES = {"tmax_f": "Int16", "tmin_f": "Int16", "revised_tmax_f": "Int16", "revised_tmin_f": "Int16",
                 "tmax_c": "float32", "tmin_c": "float32", "is_final": "bool", "revised": "bool"}
_DAILY_DTYPES = {"lead_day": "int8", "n_samples": "int8", "tmax_sampled_c": "float32", "tmin_sampled_c": "float32",
                 "tmax_native_c": "float32", "tmin_native_c": "float32"}


def _cast(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Narrow column dtypes per DESIGN §3 (tolerant: skips columns that cannot be cast)."""
    for col, dt in dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dt)
            except (TypeError, ValueError):
                pass
    return df


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pq.read_table(path).to_pandas()


def _write(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(table, _ensure(tmp), compression="zstd")
    tmp.replace(path)


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
        merged = _upsert(_read(path, FORECAST_VALUE_COLUMNS), part, FORECAST_VALUE_KEY, "missing_reason")
        _write(_cast(merged, _FV_DTYPES), path)
        out[str(path.relative_to(DATA_DIR))] = len(merged)
    return out


def read_forecast_values(model_ids: list[str] | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    base = DATA_DIR / "forecast_values"
    if not base.exists():
        return pd.DataFrame(columns=FORECAST_VALUE_COLUMNS)
    frames = []
    for mdir in sorted(base.glob("model_id=*")):
        mid = mdir.name.split("=", 1)[1]
        if model_ids and mid not in model_ids:
            continue
        for f in sorted(mdir.glob("year_month=*.parquet")):
            ym = f.stem.split("=", 1)[1]
            if start and ym < start[:7]:
                continue
            if end and ym > end[:7]:
                continue
            frames.append(pq.read_table(f).to_pandas())
    if not frames:
        return pd.DataFrame(columns=FORECAST_VALUE_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if start:
        df = df[df["init_time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["init_time"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def existing_inits(model_id: str, min_valid: int | None = None) -> set[pd.Timestamp]:
    """Init times whose run is *complete enough* to skip: at least `min_valid` distinct valid times with a
    present t2 value (default: max_h/step_h from models.yaml, i.e. every 6-hourly step out to 240 h).
    Partial runs (e.g. a 503 on one step) are therefore retried by later fetch-latest/backfill passes."""
    from .config import model_by_id

    df = read_forecast_values([model_id])
    if df.empty:
        return set()
    if min_valid is None:
        try:
            m = model_by_id(model_id)
            min_valid = m.max_h // m.step_h
        except KeyError:
            min_valid = 1
    ok = df[(df["variable"] == "t2") & (df["missing_reason"] == "")]
    n_valid = ok.groupby("init_time")["valid_time"].nunique()
    return set(n_valid[n_valid >= min_valid].index)


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
        idx = first.index.intersection(latest.index)
        rev_cols = ["revised", "revised_tmax_f", "revised_tmin_f"]
        first.loc[idx, rev_cols] = latest.loc[idx, rev_cols]

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


def read_truth(years: list[int] | None = None) -> pd.DataFrame:
    base = DATA_DIR / "truth_daily"
    if not base.exists():
        return pd.DataFrame(columns=TRUTH_COLUMNS)
    frames = [pq.read_table(f).to_pandas() for f in sorted(base.glob("year=*.parquet"))
              if not years or int(f.stem.split("=")[1]) in years]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRUTH_COLUMNS)


# ---------------- daily_forecasts ----------------

def daily_path(model_id: str, year: int) -> Path:
    return DATA_DIR / "daily_forecasts" / f"model_id={model_id}" / f"year={year}.parquet"


def write_daily(df: pd.DataFrame) -> dict[str, int]:
    """Daily tables are fully re-derived; overwrite per (model, year)."""
    if df.empty:
        return {}
    df = df[DAILY_COLUMNS].copy()
    out = {}
    years = pd.to_datetime(df["climo_date"]).dt.year
    for (model_id, year), part in df.groupby([df["model_id"], years]):
        path = daily_path(model_id, int(year))
        _write(_cast(part.reset_index(drop=True), _DAILY_DTYPES), path)
        out[str(path.relative_to(DATA_DIR))] = len(part)
    return out


def read_daily(model_ids: list[str] | None = None) -> pd.DataFrame:
    base = DATA_DIR / "daily_forecasts"
    if not base.exists():
        return pd.DataFrame(columns=DAILY_COLUMNS)
    frames = []
    for mdir in sorted(base.glob("model_id=*")):
        mid = mdir.name.split("=", 1)[1]
        if model_ids and mid not in model_ids:
            continue
        frames += [pq.read_table(f).to_pandas() for f in sorted(mdir.glob("year=*.parquet"))]
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
