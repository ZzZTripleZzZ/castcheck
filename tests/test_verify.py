"""Tests for castcheck.verify (METHODOLOGY §4–§6) on synthetic daily forecasts and truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from castcheck.store import DAILY_COLUMNS, TRUTH_COLUMNS
from castcheck.verify import (
    ALL_STATIONS,
    PAIRWISE_COLUMNS,
    PERSISTENCE_ID,
    SCORE_COLUMNS,
    persistence_daily,
    score,
    select_truth,
)

STATIONS = ("KAAA", "KBBB")
DATES = pd.date_range("2026-06-01", "2026-08-29", freq="D")  # 90 days
#: constant error in °C added to the observation, per model and station
OFFSETS = {
    "exact": {"KAAA": 0.0, "KBBB": 0.0},
    "warm1": {"KAAA": 1.0, "KBBB": 1.0},
    "split": {"KAAA": 1.0, "KBBB": 3.0},
    "cold3": {"KAAA": -3.0, "KBBB": -3.0},
}


def make_truth(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for st in STATIONS:
        tmax = 25 + 6 * np.sin(np.arange(len(DATES)) / 9.0) + rng.normal(0, 1.5, len(DATES))
        for i, d in enumerate(DATES):
            rows.append({
                "station_id": st, "climo_date": d.date(), "source": "CLI",
                "tmax_f": None, "tmin_f": None,
                "tmax_c": float(tmax[i]), "tmin_c": float(tmax[i] - 10.0),
                "issuance_time": pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=10),
                "is_final": True, "revised": False, "revised_tmax_f": None,
                "revised_tmin_f": None, "qc_flag": "", "product_id": f"p{i}",
                "schema_version": "0.1", "methodology_version": "0.1",
            })
    return pd.DataFrame(rows)[TRUTH_COLUMNS]


def make_daily(truth: pd.DataFrame, noise_sd: float = 2.0, seed: int = 11) -> pd.DataFrame:
    """One init hour, one method, lead days 1 and 3; four models with known error structure."""
    rng = np.random.default_rng(seed)
    obs = {(r.station_id, r.climo_date): (r.tmax_c, r.tmin_c) for r in truth.itertuples()}
    rows = []
    for st in STATIONS:
        for model_id, off in OFFSETS.items():
            for lead in (1, 3):
                for d in DATES:
                    tmax, tmin = obs[(st, d.date())]
                    rows.append({
                        "model_id": model_id, "model_version": "v1",
                        "init_time": pd.Timestamp(d, tz="UTC") - pd.Timedelta(days=lead),
                        "station_id": st, "climo_date": d.date(), "lead_day": lead,
                        "method": "bilinear",
                        "tmax_sampled_c": tmax + off[st], "tmin_sampled_c": tmin + off[st],
                        "n_samples": 4, "tmax_native_c": np.nan, "tmin_native_c": np.nan,
                        "missing_reason": "", "schema_version": "0.1", "methodology_version": "0.1",
                    })
        for lead in (1, 3):
            for d in DATES:
                tmax, tmin = obs[(st, d.date())]
                e = rng.normal(0, noise_sd)
                rows.append({
                    "model_id": "noisy", "model_version": "v1",
                    "init_time": pd.Timestamp(d, tz="UTC") - pd.Timedelta(days=lead),
                    "station_id": st, "climo_date": d.date(), "lead_day": lead,
                    "method": "bilinear",
                    "tmax_sampled_c": tmax + e, "tmin_sampled_c": tmin + e,
                    "n_samples": 4, "tmax_native_c": np.nan, "tmin_native_c": np.nan,
                    "missing_reason": "", "schema_version": "0.1", "methodology_version": "0.1",
                })
    return pd.DataFrame(rows)[DAILY_COLUMNS]


@pytest.fixture(scope="module")
def truth():
    return make_truth()


@pytest.fixture(scope="module")
def daily(truth):
    return make_daily(truth)


@pytest.fixture(scope="module")
def scored(daily, truth):
    return score(daily, truth, windows=(30, 90, None), n_boot=500, seed=0)


def pick(scores, **kw):
    m = pd.Series(True, index=scores.index)
    for k, v in kw.items():
        m &= scores[k] == v
    got = scores[m]
    assert len(got) == 1, f"expected exactly one row for {kw}, got {len(got)}"
    return got.iloc[0]


# ---------------------------------------------------------------------------------- truth

def test_select_truth_prefers_final_cli_then_cf6_then_obs():
    rows = []
    for source, is_final, value in (("OBS", False, 10.0), ("CF6", False, 11.0), ("CLI", True, 12.0)):
        rows.append({"station_id": "KAAA", "climo_date": pd.Timestamp("2026-06-01").date(),
                     "source": source, "is_final": is_final, "tmax_c": value, "tmin_c": value - 5,
                     "qc_flag": "", "issuance_time": pd.Timestamp("2026-06-02", tz="UTC")})
    t = pd.DataFrame(rows)
    sel = select_truth(t)
    assert pick(sel, variable="tmax")["obs_c"] == 12.0
    assert pick(sel, variable="tmax")["truth_source"] == "CLI"
    # drop the CLI row → CF6 wins; drop CF6 too → OBS wins and is flagged
    sel2 = select_truth(t[t["source"] != "CLI"])
    assert pick(sel2, variable="tmax")["obs_c"] == 11.0
    sel3 = select_truth(t[t["source"] == "OBS"])
    assert pick(sel3, variable="tmax")["obs_c"] == 10.0
    assert pick(sel3, variable="tmax")["qc_flag"] == "obs_fallback"


# ---------------------------------------------------------------------------- persistence

def test_persistence_is_the_observation_lead_days_earlier(truth):
    pers = persistence_daily(truth)
    assert list(pers.columns) == DAILY_COLUMNS
    assert set(pers["model_id"]) == {PERSISTENCE_ID}
    assert 0 not in set(pers["lead_day"])  # lead 0 would be the target day itself
    obs = {(r.station_id, r.climo_date): r.tmax_c for r in truth.itertuples()}
    sub = pers[(pers["station_id"] == "KAAA") & (pers["lead_day"] == 3)
               & (pers["method"] == "bilinear") & (pers["init_time"].dt.hour == 0)]
    assert len(sub) == len(DATES) - 3
    for _, r in sub.head(20).iterrows():
        source_day = (pd.Timestamp(r["climo_date"]) - pd.Timedelta(days=3)).date()
        assert r["tmax_sampled_c"] == pytest.approx(obs[("KAAA", source_day)], abs=1e-5)
    # the baseline is emitted for both inits and both methods so it lines up with every model group
    assert set(pers["method"]) == {"bilinear", "nearest"}
    assert set(pers["init_time"].dt.hour) == {0, 12}


# --------------------------------------------------------------------------------- scores

def test_score_schema(scored):
    scores, pairwise = scored
    assert list(scores.columns) == SCORE_COLUMNS
    assert list(pairwise.columns) == PAIRWISE_COLUMNS
    assert set(scores["window"]) == {"30d", "90d", "all"}
    assert set(scores["variable"]) == {"tmax", "tmin"}
    assert PERSISTENCE_ID in set(scores["model_id"])


def test_known_mae_bias_rmse_and_hit_rates(scored):
    scores, _ = scored
    r = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="tmax",
             method="bilinear", init_hour=0, window="90d")
    assert r["n"] == len(DATES)
    assert r["mae"] == pytest.approx(1.0, abs=1e-5)
    assert r["bias"] == pytest.approx(1.0, abs=1e-5)
    assert r["rmse"] == pytest.approx(1.0, abs=1e-5)
    # a constant 1 °C error is 1.8 °F: outside ±1 °F, inside ±2 °F and ±3 °F
    assert r["hit1f"] == pytest.approx(0.0)
    assert r["hit2f"] == pytest.approx(1.0)
    assert r["hit3f"] == pytest.approx(1.0)
    assert str(r["period_start"]) == "2026-06-01" and str(r["period_end"]) == "2026-08-29"

    cold = pick(scores, station_id="KAAA", model_id="cold3", lead_day=1, variable="tmax",
                method="bilinear", init_hour=0, window="90d")
    assert cold["mae"] == pytest.approx(3.0, abs=1e-5)
    assert cold["bias"] == pytest.approx(-3.0, abs=1e-5)


def test_window_lengths(scored):
    scores, _ = scored
    r30 = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="tmax",
               method="bilinear", init_hour=0, window="30d")
    assert r30["n"] == 30
    assert str(r30["period_start"]) == "2026-07-31"
    r_all = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="tmax",
                 method="bilinear", init_hour=0, window="all")
    assert r_all["n"] == len(DATES)


def test_skill_versus_persistence(scored):
    scores, _ = scored
    exact = pick(scores, station_id="KAAA", model_id="exact", lead_day=1, variable="tmax",
                 method="bilinear", init_hour=0, window="90d")
    assert exact["mae"] == pytest.approx(0.0, abs=1e-9)
    assert exact["skill_persistence"] == pytest.approx(1.0, abs=1e-6)
    pers = pick(scores, station_id="KAAA", model_id=PERSISTENCE_ID, lead_day=1, variable="tmax",
                method="bilinear", init_hour=0, window="90d")
    assert np.isnan(pers["skill_persistence"])  # the baseline has no skill against itself
    assert pers["mae"] > 0
    warm = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="tmax",
                method="bilinear", init_hour=0, window="90d")
    assert warm["skill_persistence"] == pytest.approx(1.0 - warm["mae"] / pers["mae"], abs=0.05)


def test_bootstrap_interval_brackets_the_truth_and_is_reproducible(daily, truth):
    scores, _ = score(daily, truth, windows=(None,), n_boot=1000, seed=0)
    r = pick(scores, station_id="KAAA", model_id="noisy", lead_day=1, variable="tmax",
             method="bilinear", init_hour=0, window="all")
    # errors are N(0, 2): population MAE = 2·sqrt(2/π) ≈ 1.596, population bias = 0
    population_mae = 2.0 * np.sqrt(2.0 / np.pi)
    assert r["mae_ci_low"] < population_mae < r["mae_ci_high"]
    assert r["bias_ci_low"] < 0.0 < r["bias_ci_high"]
    assert r["mae_ci_low"] < r["mae"] < r["mae_ci_high"]
    again, _ = score(daily, truth, windows=(None,), n_boot=1000, seed=0)
    r2 = pick(again, station_id="KAAA", model_id="noisy", lead_day=1, variable="tmax",
              method="bilinear", init_hour=0, window="all")
    assert r2["mae_ci_low"] == r["mae_ci_low"] and r2["mae_ci_high"] == r["mae_ci_high"]
    other, _ = score(daily, truth, windows=(None,), n_boot=1000, seed=1)
    r3 = pick(other, station_id="KAAA", model_id="noisy", lead_day=1, variable="tmax",
              method="bilinear", init_hour=0, window="all")
    assert r3["mae_ci_low"] != r["mae_ci_low"]


def test_constant_error_gives_a_degenerate_interval(scored):
    scores, _ = scored
    r = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="tmax",
             method="bilinear", init_hour=0, window="90d")
    assert r["mae_ci_low"] == pytest.approx(1.0, abs=1e-5)
    assert r["mae_ci_high"] == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------------------- ALL rows

def test_all_station_rows_average_the_daily_station_errors(scored):
    scores, _ = scored
    a = pick(scores, station_id=ALL_STATIONS, model_id="split", lead_day=1, variable="tmax",
             method="bilinear", init_hour=0, window="90d")
    # KAAA is +1 °C every day and KBBB is +3 °C, so the daily cross-station mean error is +2
    assert a["mae"] == pytest.approx(2.0, abs=1e-5)
    assert a["bias"] == pytest.approx(2.0, abs=1e-5)
    assert a["n"] == len(DATES)
    # ±3 °F = 1.667 °C: KAAA is inside, KBBB is not, so the daily hit rate is 0.5
    assert a["hit3f"] == pytest.approx(0.5, abs=1e-5)
    assert ALL_STATIONS in set(scores["station_id"])
    assert set(scores.loc[scores["station_id"] == ALL_STATIONS, "model_id"]) == \
        set(scores.loc[scores["station_id"] == "KAAA", "model_id"])


# -------------------------------------------------------------------------------- pairwise

def test_pairwise_detects_a_clear_difference(scored):
    _, pairwise = scored
    row = pairwise[(pairwise["station_id"] == "KAAA") & (pairwise["lead_day"] == 1)
                   & (pairwise["variable"] == "tmax") & (pairwise["window"] == "90d")
                   & (pairwise["method"] == "bilinear")
                   & (pairwise["model_a"] == "cold3") & (pairwise["model_b"] == "warm1")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["n_common"] == len(DATES)
    assert r["mae_diff"] == pytest.approx(2.0, abs=1e-5)  # 3.0 − 1.0
    assert r["ci_low"] > 0
    assert bool(r["significant"]) is True


def test_pairwise_is_not_significant_when_models_are_identical(truth):
    daily = make_daily(truth)
    twin = daily[daily["model_id"] == "noisy"].copy()
    twin["model_id"] = "noisy_twin"
    _, pairwise = score(pd.concat([daily, twin], ignore_index=True), truth,
                        windows=(90,), n_boot=500, seed=0)
    row = pairwise[(pairwise["station_id"] == "KAAA") & (pairwise["lead_day"] == 1)
                   & (pairwise["variable"] == "tmax")
                   & (pairwise["model_a"] == "noisy") & (pairwise["model_b"] == "noisy_twin")]
    assert len(row) == 1
    assert row.iloc[0]["mae_diff"] == pytest.approx(0.0, abs=1e-9)
    assert bool(row.iloc[0]["significant"]) is False


def test_pairwise_uses_only_common_days(truth):
    daily = make_daily(truth)
    short = daily[(daily["model_id"] == "warm1")].copy()
    short["model_id"] = "warm1_short"
    keep = pd.to_datetime(short["climo_date"]) >= pd.Timestamp("2026-08-01")
    short = short[keep]
    _, pairwise = score(pd.concat([daily, short], ignore_index=True), truth,
                        windows=(None,), n_boot=200, seed=0)
    row = pairwise[(pairwise["station_id"] == "KAAA") & (pairwise["lead_day"] == 1)
                   & (pairwise["variable"] == "tmax")
                   & (pairwise["model_a"] == "cold3") & (pairwise["model_b"] == "warm1_short")]
    assert len(row) == 1
    assert row.iloc[0]["n_common"] == 29  # 2026-08-01 .. 2026-08-29


# --------------------------------------------------------------------------------- corners

def test_small_samples_are_scored_but_reported_with_their_n(truth):
    daily = make_daily(truth)
    few = pd.to_datetime(daily["climo_date"]) >= pd.Timestamp("2026-08-20")
    scores, _ = score(daily[few], truth, windows=(None,), n_boot=200, seed=0)
    r = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="tmax",
             method="bilinear", init_hour=0, window="all")
    assert r["n"] == 10
    assert r["mae"] == pytest.approx(1.0, abs=1e-5)


def test_empty_inputs_return_empty_tables():
    empty_daily = pd.DataFrame(columns=DAILY_COLUMNS)
    empty_truth = pd.DataFrame(columns=TRUTH_COLUMNS)
    scores, pairwise = score(empty_daily, empty_truth)
    assert scores.empty and pairwise.empty
    assert list(scores.columns) == SCORE_COLUMNS
    assert list(pairwise.columns) == PAIRWISE_COLUMNS
