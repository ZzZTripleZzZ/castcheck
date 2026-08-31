"""Tests for castcheck.derive (METHODOLOGY §2.3–2.5) on synthetic forecast_values."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from castcheck.config import ModelSpec, Station
from castcheck.derive import (
    INSTANT_ERROR_COLUMNS,
    daily_columns,
    daily_from_values,
    extreme_kind,
    instant_errors,
    native_overhang_hours,
    observed_sampled_extremes,
)
from castcheck.sources.base import FORECAST_VALUE_COLUMNS, make_rows
from castcheck.store import DAILY_COLUMNS

# A −6 h station: its climatological day [06Z, 06Z+24h) is a whole number of 3 h and 6 h buckets,
# so native extremes can tile it exactly.
CHI = Station(id="KORD", name="Chicago O'Hare", cli_pil="CLIORD", tz="America/Chicago",
              std_offset_h=-6, lat=41.98, lon=-87.90, elev_m=201.0)
# A −5 h station: no bucket set tiles its day exactly, so the native run overhangs the day by one
# bucket length in total (METHODOLOGY §2.4).
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
    # last covered day of the run: its buckets stop at the horizon, so the day is not covered
    daily = daily_from_values(build_values(CHI, IFS), [CHI], [IFS])
    last = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 2)].iloc[0]
    assert np.isnan(last["tmax_native_c"]) and np.isnan(last["tmin_native_c"])
    assert last["n_samples"] == 4  # the sampled extremes are still fine


def test_native_overhang_is_a_function_of_offset_and_bucket_length():
    # METHODOLOGY §2.4: 3 h and 6 h buckets are anchored to 00 UTC, so only a −6 h station is tiled
    # exactly; −5/−7/−8 h stations overhang their day by one bucket length in total.
    assert native_overhang_hours(-6, 3) == 0.0
    assert native_overhang_hours(-6, 6) == 0.0
    for off in (-5, -7, -8):
        assert native_overhang_hours(off, 3) == 3.0
        assert native_overhang_hours(off, 6) == 6.0


def test_native_extremes_use_one_crossing_bucket_at_each_end_for_a_non_aligned_station():
    """A −5 h station is covered by 03Z..06Z+24h, i.e. 21 h inside the day plus a 2 h and a 1 h
    overhang (METHODOLOGY §2.4 v0.2).  Before v0.2 these 15 stations all got NaN."""
    daily = daily_from_values(build_values(NYC, IFS), [NYC], [IFS])
    row = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 1)].iloc[0]
    # day 1 of a −5 h station = 2026-08-02 05Z .. 2026-08-03 05Z; the covering run is
    # (03Z, 06Z] on 08-02 through (03Z, 06Z] on 08-03
    hours = [datetime(2026, 8, 2, h, tzinfo=UTC) for h in range(6, 24, 3)]
    hours += [datetime(2026, 8, 3, h, tzinfo=UTC) for h in (0, 3, 6)]
    vals = [t2_curve(v) for v in hours]
    assert row["tmax_native_c"] == pytest.approx(max(vals) + 1.0, abs=1e-5)
    assert row["tmin_native_c"] == pytest.approx(min(vals) - 1.0, abs=1e-5)
    assert row["tmax_native_c"] > row["tmax_sampled_c"]
    # the run may not overhang by more than one bucket at each end
    assert native_overhang_hours(NYC.std_offset_h, 3) == 3.0


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


# ------------------------------------------------------------------ v0.3: observed instants


def truth_instant_rows(station: Station, *, skip: set[datetime] | None = None,
                       offset: float = 0.0) -> pd.DataFrame:
    """Observed 2 m temperature at every 6-hourly instant the fixture forecasts cover."""
    skip = skip or set()
    rows = []
    t = INIT - timedelta(days=1)
    while t <= INIT + timedelta(hours=96):
        if t.hour % 6 == 0 and t not in skip:
            rows.append({"station_id": station.id, "valid_time": t,
                         "temp_c": t2_curve(t) + offset, "obs_time": t, "source": "ASOS_IEM",
                         "n_reports": 1, "qc_flag": "", "schema_version": "0.3",
                         "methodology_version": "0.3"})
        t += timedelta(hours=1)
    return pd.DataFrame(rows)


def test_observed_sampled_extremes_are_the_four_instants_of_the_day():
    ti = truth_instant_rows(CHI)
    obs = observed_sampled_extremes(ti, [CHI])
    row = obs[obs["climo_date"] == date(2026, 8, 2)].iloc[0]
    samples = [t2_curve(datetime(2026, 8, 2, h, tzinfo=UTC)) for h in (6, 12, 18)]
    samples.append(t2_curve(datetime(2026, 8, 3, 0, tzinfo=UTC)))
    assert row["n_obs_samples"] == 4
    assert row["tmax_obs_s_c"] == pytest.approx(max(samples), abs=1e-5)
    assert row["tmin_obs_s_c"] == pytest.approx(min(samples), abs=1e-5)


def test_observed_sampled_extremes_need_all_four_instants():
    ti = truth_instant_rows(CHI, skip={datetime(2026, 8, 2, 18, tzinfo=UTC)})
    obs = observed_sampled_extremes(ti, [CHI])
    row = obs[obs["climo_date"] == date(2026, 8, 2)].iloc[0]
    assert row["n_obs_samples"] == 3
    assert np.isnan(row["tmax_obs_s_c"]) and np.isnan(row["tmin_obs_s_c"])


def test_daily_carries_the_observed_sampled_extremes_and_the_native_overhang():
    ti = truth_instant_rows(CHI)
    daily = daily_from_values(build_values(CHI, IFS), [CHI], [IFS], truth_instant=ti)
    assert list(daily.columns) == daily_columns()
    row = daily[(daily["method"] == "bilinear") & (daily["lead_day"] == 1)].iloc[0]
    assert row["n_obs_samples"] == 4
    # the fixture's forecast is the same curve as the observation, so the extremes coincide
    assert row["tmax_obs_s_c"] == pytest.approx(row["tmax_sampled_c"], abs=1e-5)
    # −6 h station with 3 h buckets: the native run tiles the day exactly
    assert row["native_overhang_h"] == pytest.approx(0.0)
    nyc = daily_from_values(build_values(NYC, IFS), [NYC], [IFS],
                            truth_instant=truth_instant_rows(NYC))
    nrow = nyc[(nyc["method"] == "bilinear") & (nyc["lead_day"] == 1)].iloc[0]
    assert nrow["native_overhang_h"] == pytest.approx(native_overhang_hours(-5, 3))


def test_daily_without_truth_instant_leaves_the_observed_columns_empty():
    daily = daily_from_values(build_values(CHI, IFS), [CHI], [IFS])
    assert daily["tmax_obs_s_c"].isna().all()
    assert (daily["n_obs_samples"] == 0).all()


# --------------------------------------------------------------------- v0.3: instant errors


def test_instant_errors_are_one_row_per_common_instant():
    ti = truth_instant_rows(CHI, offset=-1.0)   # the observation is 1 °C below the forecast
    inst = instant_errors(build_values(CHI, IFS), ti, [CHI], [IFS])
    assert list(inst.columns) == INSTANT_ERROR_COLUMNS
    assert set(inst["valid_hour_utc"]) == {0, 6, 12, 18}
    bil = inst[inst["method"] == "bilinear"]
    assert bil["err_c"].round(4).eq(1.0).all()           # forecast − observation
    # the `nearest` variant carries the +0.5 offset the fixture puts on it
    assert inst[inst["method"] == "nearest"]["err_c"].round(4).eq(1.5).all()
    # every covered climatological day contributes exactly four instants
    per_day = bil.groupby(["lead_day", "climo_date"]).size()
    assert set(per_day) == {4}
    assert sorted(bil["lead_day"].unique()) == [0, 1, 2]
    row = bil.iloc[0]
    assert row["lead_h"] == int((row["valid_time"] - row["init_time"]).total_seconds() // 3600)
    assert row["model_version"] == "ifs-cy50r1"


def test_instant_errors_drop_instants_without_an_observation():
    ti = truth_instant_rows(CHI, skip={datetime(2026, 8, 2, 18, tzinfo=UTC)})
    inst = instant_errors(build_values(CHI, IFS), ti, [CHI], [IFS])
    bil = inst[(inst["method"] == "bilinear") & (inst["climo_date"] == date(2026, 8, 2))]
    assert len(bil) == 3
    assert 18 not in set(bil["valid_hour_utc"])


def test_instant_errors_are_empty_without_truth():
    empty = pd.DataFrame(columns=["station_id", "valid_time", "temp_c", "qc_flag"])
    out = instant_errors(build_values(CHI, IFS), empty, [CHI], [IFS])
    assert out.empty
    assert list(out.columns) == INSTANT_ERROR_COLUMNS
