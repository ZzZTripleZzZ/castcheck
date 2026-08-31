"""Union-merge two versions of a data shard (used to resolve git conflicts between writers).

Two independent writers (GitHub Actions and the local launchd mirror) may both add rows to the same
monthly shard. Picking either side would silently drop the other side's rows, so conflicted shards
are merged with the *same* semantics the store uses for upserts:

- ``forecast_values``: one row per key; a present value beats a missing one, later fetch beats earlier.
- ``truth_daily``: first-final policy per (station, day, source); revisions/QC flags are unioned.
- ``truth_instant``: one row per (station, instant); a value beats a gap, the ASOS archive beats the
  api.weather.gov stop-gap.
- ``daily_forecasts``: one row per DAILY_KEY, later side wins. It used to be safe to keep whichever
  file was newer, because ``derive`` rewrote every shard from scratch; once ``derive`` takes a date
  window the newer file is only newer *for that window*, so the sides have to be unioned.
- ``scores``: fully re-derived from the tables above — keep the later-computed file (methodology version, computed_at), the next
  ``castcheck verify`` regenerates it anyway.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .sources.base import FORECAST_VALUE_COLUMNS, FORECAST_VALUE_KEY
from .store import (
    _DAILY_DTYPES,
    _FV_DTYPES,
    _TRUTH_DTYPES,
    _TRUTH_INSTANT_DTYPES,
    DAILY_COLUMNS,
    DAILY_KEY,
    TRUTH_COLUMNS,
    TRUTH_INSTANT_COLUMNS,
    _apply_first_final,
    _cast,
    _conform,
    _upsert,
    _write,
    resolve_truth_instant,
)


def merge_frames(kind: str, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if kind == "forecast_values":
        a, b = a[FORECAST_VALUE_COLUMNS], b[FORECAST_VALUE_COLUMNS]
        for df in (a, b):
            for c in ("init_time", "valid_time", "fetched_at"):
                df[c] = pd.to_datetime(df[c], utc=True)
        return _cast(_upsert(a, b, FORECAST_VALUE_KEY, "missing_reason"), _FV_DTYPES)
    if kind == "truth_daily":
        a, b = a[TRUTH_COLUMNS].copy(), b[TRUTH_COLUMNS].copy()
        for df in (a, b):
            df["climo_date"] = pd.to_datetime(df["climo_date"]).dt.date
            df["issuance_time"] = pd.to_datetime(df["issuance_time"], utc=True)
        return _cast(_apply_first_final(pd.concat([a, b], ignore_index=True)), _TRUTH_DTYPES)
    if kind == "truth_instant":
        a, b = a[TRUTH_INSTANT_COLUMNS].copy(), b[TRUTH_INSTANT_COLUMNS].copy()
        for df in (a, b):
            for c in ("valid_time", "obs_time"):
                df[c] = pd.to_datetime(df[c], utc=True)
        return _cast(resolve_truth_instant(pd.concat([a, b], ignore_index=True)), _TRUTH_INSTANT_DTYPES)
    if kind == "daily_forecasts":
        a, b = _conform(a.copy(), DAILY_COLUMNS), _conform(b.copy(), DAILY_COLUMNS)
        for df in (a, b):
            df["climo_date"] = pd.to_datetime(df["climo_date"]).dt.date
            df["init_time"] = pd.to_datetime(df["init_time"], utc=True)
        merged = (pd.concat([a, b], ignore_index=True)
                  .drop_duplicates(subset=DAILY_KEY, keep="last")
                  .sort_values(DAILY_KEY).reset_index(drop=True))
        return _cast(merged, _DAILY_DTYPES)
    raise ValueError(kind)


def kind_of(path: Path) -> str:
    parts = path.as_posix()
    for k in ("forecast_values", "truth_daily", "truth_instant", "daily_forecasts", "scores"):
        if f"/{k}/" in parts or parts.startswith(f"{k}/"):
            return k
    raise ValueError(f"unknown shard kind: {path}")


def merge_files(ours: Path, theirs: Path, out: Path) -> str:
    kind = kind_of(out)
    if kind == "scores":
        # Fully re-derived: keep the later-computed table (methodology version, then computed_at), not mtime.
        def _stamp(p: Path) -> tuple:
            names = pq.read_schema(p).names
            cols = [c for c in ("methodology_version", "computed_at") if c in names]
            d = pq.read_table(p, columns=cols).to_pandas() if cols else None
            mv = str(d["methodology_version"].iloc[0]) if d is not None and "methodology_version" in d and len(d) else ""
            ca = str(d["computed_at"].max()) if d is not None and "computed_at" in d and len(d) else ""
            return (mv, ca, p.stat().st_mtime)
        newer = ours if _stamp(ours) >= _stamp(theirs) else theirs
        out.write_bytes(newer.read_bytes())
        return f"{out}: derived table, kept newer ({newer.name})"
    a = pq.read_table(ours).to_pandas()
    b = pq.read_table(theirs).to_pandas()
    merged = merge_frames(kind, a, b)
    _write(merged, out)
    return f"{out}: {kind} union {len(a)} + {len(b)} -> {len(merged)} rows"


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: python -m castcheck.merge <ours.parquet> <theirs.parquet> <out.parquet>", file=sys.stderr)
        return 2
    print(merge_files(Path(argv[1]), Path(argv[2]), Path(argv[3])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
