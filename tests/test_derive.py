"""Tests for castcheck.derive (METHODOLOGY §2.3–2.5) on synthetic forecast_values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from castcheck.config import ModelSpec, Station
from castcheck.derive import daily_from_values, extreme_kind
from castcheck.sources.base import FORECAST_VALUE_COLUMNS, make_rows
from castcheck.store import DAILY_COLUMNS

# A −6 h station: its climatological day [06Z, 06Z+24h) is a whole number of 3 h and 6 h buckets,
# so native extremes can tile it exactly.
CHI = Station(id="KORD", name="Chicago O'Hare", cli_pil="CLIORD", tz="America/Chicago",
              std_offset_h=-6, lat=41.98, lon=-87.90, elev_m=201.0)
# A −5 h station: 6 h buckets can never tile its day, so native extremes must be NaN there.
NYC = Station(id="KNYC", name="New York Central Park", cli_pil="CLINYC", tz="America/New_York",
              std_offset_h=-5, lat=40.78, lon=-73.97, elev_m=47.0)

IFS = ModelSpec(model_id="ifs_hres", family="ECMWF IFS HRES", source="ecmwf", product="oper",
                init_field=None, inits=(0, 12), step_h=3, max_h=72,
                native_extremes=("mx2t3", "mn2t3"))
AI = ModelSpec(model_id="graphcast_ifs", family="GraphCast", source="aiwp", product="GRAP",
               init_field="IFS", inits=(0, 12), step_h=6, max_h=72, native_extremes=())

INIT = datetime(2026, 8, 1, 0, tzinfo=UTC)


def t2_curve(valid: datetime) -> float:
    """A deterministic diurnal cycle: warmest at 18Z, coldest at 06Z."""
    return 20.0 + 5.0 * np.sin(2 * np.pi * (valid.hour - 12) / 24.0)


def build_values(station: Station, model: ModelSpec, *, native: bool = True,
                 skip_valid: set[datetime] | None = None) -> pd.DataFrame:
    skip_valid = skip_valid or set()
    frames = []
    for h in range(model.step_h, model.max_h + 1, model.step_h):
        valid = INIT + timedelta(hours=h)
        if valid in skip_valid:
            continue
        v = t2_curve(valid)
        if valid.hour % 6 == 0:
            frames.append(make_rows(model=model, model_version="ifs-cy50r1", init_time=INIT,
                                    valid_time=valid, lead_h=h, variable="t2", bucket_h=0,
                                    source_url="test://t2", values={station.id: (v, v + 0.5)},
                                    stations=[station]))
        if native and model.native_extremes:
            frames.append(make_rows(model=model, model_version="ifs-cy50r1", init_time=INIT,
                                    valid_time=valid, lead_h=h, variable="mx2t3", bucket_h=3,
                                    source_url="test://mx", values={station.id: (v + 1.0, v + 1.5)},
                                    stations=[station]))
            frames.append(make_rows(model=model, model_version="ifs-cy50r1", init_time=INIT,
                                    valid_time=valid, lead_h=h, variable="mn2t3", bucket_h=3,
                                    source_url="test://mn", values={station.id: (v - 1.0, v - 0.5)},
                                    stations=[station]))
    return pd.concat(frames, ignore_index=True)


def test_extreme_kind_recognises_every_adapter_spelling():
    assert extreme_kind("mx2t3") == "max"
    assert extreme_kind("mx2t6") == "max"
    assert extreme_kind("tmax6") == "max"
    assert extreme_kind("TMAX") == "max"
    assert extreme_kind("mn2t6") == "min"
    assert extreme_kind("tmin6") == "min"
    assert extreme_kind("t2") is None


def test_schema_and_lead_days():
    daily = daily_from_values(build_values(CHI, IFS), [CHI], [IFS])
    assert list(daily.columns) == DAILY_COLUMNS
    assert not daily.empty
    # a 00Z run with a 72 h horizon covers the −6 h station's day 0, 1 and 2
    assert sorted(daily["lead_day"].unique().tolist()) == [0, 1, 2]
    for _, r in daily.iterrows():
        assert (r["climo_date"] - INIT.date()).days == r["lead_day"]
    assert set(daily["method"]) == {"bilinear", "nearest"}
    assert set(daily["model_version"]) == {"ifs-cy50r1"}


def test_sampled_extremes_are_the_four_common_samples():
    daily = daily_from_values(build_values(CHI, IFS), [CHI], [IFS])
    row = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 1)].iloc[0]
    # day 1 of a −6 h station = 2026-08-02 06Z .. 2026-08-03 06Z → samples 06/12/18/00Z
    samples = [t2_curve(datetime(2026, 8, 2, h, tzinfo=UTC)) for h in (6, 12, 18)]
    samples.append(t2_curve(datetime(2026, 8, 3, 0, tzinfo=UTC)))
    assert row["n_samples"] == 4
    assert row["tmax_sampled_c"] == pytest.approx(max(samples), abs=1e-5)
    assert row["tmin_sampled_c"] == pytest.approx(min(samples), abs=1e-5)
    assert row["missing_reason"] == ""
    # the `nearest` variant carries the +0.5 offset used in the fixture
    near = daily[(daily["method"] == "nearest") & (daily["lead_day"] == 1)].iloc[0]
    assert near["tmax_sampled_c"] == pytest.approx(max(samples) + 0.5, abs=1e-5)


def test_native_extremes_when_buckets_tile_the_day():
    daily = daily_from_values(build_values(CHI, IFS), [CHI], [IFS])
    row = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 1)].iloc[0]
    hours = [datetime(2026, 8, 2, h, tzinfo=UTC) for h in range(9, 24, 3)]
    hours += [datetime(2026, 8, 3, h, tzinfo=UTC) for h in (0, 3, 6)]
    vals = [t2_curve(v) for v in hours]
    assert row["tmax_native_c"] == pytest.approx(max(vals) + 1.0, abs=1e-5)
    assert row["tmin_native_c"] == pytest.approx(min(vals) - 1.0, abs=1e-5)
    # native max must sit above the 6-hourly sampled max (METHODOLOGY §2.3 under-sampling)
    assert row["tmax_native_c"] > row["tmax_sampled_c"]


def test_native_extremes_nan_when_buckets_do_not_cover_the_whole_day():
    # last covered day of the run: its buckets stop at the horizon, so it is not tiled
    daily = daily_from_values(build_values(CHI, IFS), [CHI], [IFS])
    last = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 2)].iloc[0]
    assert np.isnan(last["tmax_native_c"]) and np.isnan(last["tmin_native_c"])
    assert last["n_samples"] == 4  # the sampled extremes are still fine

    # a −5 h station can never be tiled by buckets aligned to whole UTC hours divisible by 3
    daily_nyc = daily_from_values(build_values(NYC, IFS), [NYC], [IFS])
    assert daily_nyc["tmax_native_c"].isna().all()
    assert daily_nyc["tmin_native_c"].isna().all()


def test_models_without_native_fields_get_nan():
    daily = daily_from_values(build_values(CHI, AI, native=False), [CHI], [AI])
    assert not daily.empty
    assert daily["tmax_native_c"].isna().all()
    assert daily["tmin_native_c"].isna().all()
    assert (daily["n_samples"] == 4).all()


def test_incomplete_samples_are_explicit_missing_rows():
    skip = {datetime(2026, 8, 2, 18, tzinfo=UTC)}
    daily = daily_from_values(build_values(CHI, IFS, skip_valid=skip), [CHI], [IFS])
    row = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 1)].iloc[0]
    assert row["n_samples"] == 3
    assert np.isnan(row["tmax_sampled_c"]) and np.isnan(row["tmin_sampled_c"])
    assert row["missing_reason"] == "incomplete_samples"
    # the row still exists: nothing is silently skipped (DESIGN §0)
    assert len(daily[daily["lead_day"] == 1]) == 2
    # neighbouring days are untouched
    assert (daily[daily["lead_day"] == 0]["n_samples"] == 4).all()


def test_missing_rows_do_not_contribute_values():
    values = build_values(CHI, IFS)
    hit = (values["variable"] == "t2") & (values["valid_time"] == pd.Timestamp("2026-08-02 18:00", tz="UTC"))
    values.loc[hit, "value_c"] = np.nan
    values.loc[hit, "missing_reason"] = "no_field"
    daily = daily_from_values(values, [CHI], [IFS])
    row = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 1)].iloc[0]
    assert row["n_samples"] == 3
    assert row["missing_reason"] == "incomplete_samples"


def test_empty_input_returns_empty_table():
    empty = pd.DataFrame(columns=FORECAST_VALUE_COLUMNS)
    out = daily_from_values(empty, [CHI], [IFS])
    assert list(out.columns) == DAILY_COLUMNS
    assert out.empty
