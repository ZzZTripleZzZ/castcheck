from datetime import datetime

import numpy as np
import pandas as pd

from castcheck import merge
from castcheck.sources.base import FORECAST_VALUE_COLUMNS


def _fv(rows):
    recs = []
    for init, valid, station, value, missing, fetched in rows:
        recs.append({
            "model_id": "gfs", "model_version": "gfs-0p25",
            "init_time": pd.Timestamp(init, tz="UTC"), "valid_time": pd.Timestamp(valid, tz="UTC"),
            "lead_h": 6, "station_id": station, "variable": "t2", "bucket_h": 0, "method": "bilinear",
            "value_c": value, "missing_reason": missing, "source_url": "u",
            "fetched_at": pd.Timestamp(fetched, tz="UTC"), "schema_version": "0.1", "methodology_version": "0.2",
        })
    return pd.DataFrame.from_records(recs, columns=FORECAST_VALUE_COLUMNS)


def test_forecast_values_union_keeps_both_sides_and_prefers_present():
    ours = _fv([
        ("2026-08-30T00", "2026-08-30T06", "KNYC", 20.0, "", "2026-08-30T08"),
        ("2026-08-30T12", "2026-08-30T18", "KNYC", np.nan, "http_503", "2026-08-30T21"),
    ])
    theirs = _fv([
        ("2026-08-30T00", "2026-08-30T06", "KORD", 25.0, "", "2026-08-30T09"),   # only upstream has it
        ("2026-08-30T12", "2026-08-30T18", "KNYC", 27.5, "", "2026-08-30T22"),   # upstream got the value
    ])
    out = merge.merge_frames("forecast_values", ours, theirs)
    assert len(out) == 3
    knyc18 = out[(out.station_id == "KNYC") & (out.valid_time == pd.Timestamp("2026-08-30T18", tz="UTC"))].iloc[0]
    assert knyc18.missing_reason == "" and abs(knyc18.value_c - 27.5) < 1e-6
    assert set(out.station_id) == {"KNYC", "KORD"}


def test_truth_union_first_final_and_revision():
    base = {
        "station_id": "KNYC", "climo_date": datetime(2026, 8, 29).date(), "source": "CLI",
        "tmax_c": 30.0, "tmin_c": 20.0, "is_final": True, "revised": False,
        "revised_tmax_f": pd.NA, "revised_tmin_f": pd.NA, "qc_flag": "", "product_id": "p1",
        "schema_version": "0.1", "methodology_version": "0.2",
    }
    ours = pd.DataFrame([{**base, "tmax_f": 86, "tmin_f": 68, "issuance_time": pd.Timestamp("2026-08-30T06:30", tz="UTC")}])
    theirs = pd.DataFrame([{**base, "tmax_f": 85, "tmin_f": 68, "issuance_time": pd.Timestamp("2026-08-30T09:10", tz="UTC"),
                            "revised": True, "revised_tmax_f": 85, "product_id": "p2"}])
    out = merge.merge_frames("truth_daily", ours, theirs)
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row.tmax_f) == 86            # first-final value kept
    assert bool(row.revised) is True        # revision information unioned in


def test_kind_of_paths():
    from pathlib import Path

    assert merge.kind_of(Path("data/forecast_values/model_id=gfs/year_month=2026-08.parquet")) == "forecast_values"
    assert merge.kind_of(Path("data/truth_daily/year=2026.parquet")) == "truth_daily"
    assert merge.kind_of(Path("data/scores/latest.parquet")) == "scores"
