"""Source protocol and shared helpers for forecast adapters (DESIGN §3.1, §4)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Protocol

import pandas as pd

from .. import METHODOLOGY_VERSION, SCHEMA_VERSION
from ..config import ModelSpec, Station

FORECAST_VALUE_COLUMNS = [
    "model_id", "model_version", "init_time", "valid_time", "lead_h", "station_id", "variable",
    "bucket_h", "method", "value_c", "missing_reason", "source_url", "fetched_at",
    "schema_version", "methodology_version",
]
FORECAST_VALUE_KEY = ["model_id", "init_time", "valid_time", "station_id", "variable", "bucket_h", "method"]

K_OFFSET = 273.15


def k_to_c(kelvin: float) -> float:
    return float(kelvin) - K_OFFSET


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


@dataclass
class FetchRequest:
    model: ModelSpec
    init_time: datetime
    stations: list[Station]
    # None (default) = whatever the model offers: t2 plus its native extremes. Pass an explicit tuple of
    # storage variable names (e.g. ("t2",)) only to restrict the fetch.
    variables: tuple[str, ...] | None = None
    methods: tuple[str, ...] = ("bilinear", "nearest")


@dataclass
class FetchResult:
    rows: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def n_present(self) -> int:
        return int((self.rows["missing_reason"] == "").sum()) if len(self.rows) else 0


class Source(Protocol):
    def available_inits(self, model: ModelSpec, start: date, end: date) -> list[datetime]: ...
    def fetch_run(self, req: FetchRequest) -> FetchResult: ...


def make_rows(
    *,
    model: ModelSpec,
    model_version: str,
    init_time: datetime,
    valid_time: datetime,
    lead_h: int,
    variable: str,
    bucket_h: int,
    source_url: str,
    values: dict[str, tuple[float | None, float | None]],
    stations: list[Station],
    missing_reason: str = "",
    fetched_at: datetime | None = None,
) -> pd.DataFrame:
    """Build forecast_values rows for one (init, valid, variable) for all stations.

    `values` maps station_id -> (bilinear_c, nearest_c); a station absent from `values` (or with None)
    gets an explicit missing row with `missing_reason`.
    """
    fetched_at = fetched_at or now_utc()
    recs = []
    for s in stations:
        bl, nn = values.get(s.id, (None, None))
        for method, v in (("bilinear", bl), ("nearest", nn)):
            present = v is not None and not math.isnan(float(v))
            recs.append(
                {
                    "model_id": model.model_id,
                    "model_version": model_version,
                    "init_time": pd.Timestamp(init_time).tz_convert("UTC") if pd.Timestamp(init_time).tzinfo else pd.Timestamp(init_time, tz="UTC"),
                    "valid_time": pd.Timestamp(valid_time).tz_convert("UTC") if pd.Timestamp(valid_time).tzinfo else pd.Timestamp(valid_time, tz="UTC"),
                    "lead_h": int(lead_h),
                    "station_id": s.id,
                    "variable": variable,
                    "bucket_h": int(bucket_h),
                    "method": method,
                    "value_c": float(v) if present else float("nan"),
                    "missing_reason": "" if present else (missing_reason or "no_value"),
                    "source_url": source_url,
                    "fetched_at": pd.Timestamp(fetched_at),
                    "schema_version": SCHEMA_VERSION,
                    "methodology_version": METHODOLOGY_VERSION,
                }
            )
    return pd.DataFrame.from_records(recs, columns=FORECAST_VALUE_COLUMNS)


def missing_run_rows(
    *, model: ModelSpec, init_time: datetime, valid_times: list[datetime], stations: list[Station],
    reason: str, source_url: str, variables: tuple[str, ...] = ("t2",),
) -> pd.DataFrame:
    """Explicit missing rows for a whole run (e.g. HTTP 404 on the index)."""
    from ..climo_day import lead_hours

    frames = []
    for vt in valid_times:
        for var in variables:
            frames.append(
                make_rows(
                    model=model, model_version="unknown", init_time=init_time, valid_time=vt,
                    lead_h=lead_hours(init_time, vt), variable=var, bucket_h=0, source_url=source_url,
                    values={}, stations=stations, missing_reason=reason,
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FORECAST_VALUE_COLUMNS)
