"""Tests for castcheck.verify (METHODOLOGY §4–§6, DESIGN §10.2–§10.3) on synthetic data.

The fixture is built so that every v0.3 claim is checkable by hand:

* the observation has a diurnal cycle whose true daily maximum is 2 °C above the highest of the four
  sampled instants and whose true minimum is 1.5 °C below the lowest, so a *perfect* model scores
  0 on ``t2`` and on ``tmax_s``/``tmin_s`` and −2 / +1.5 on ``tmax_cli``/``tmin_cli``.  That is the
  A2 separation: the sampling penalty lives in the ``*_cli`` variables and nowhere else;
* ``cold18`` is 3 °C cold at 18 UTC and exact at every other instant, so the per-hour ``t2_*``
  variables must isolate it and the pooled ``t2`` must show a quarter of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from castcheck.derive import INSTANT_ERROR_COLUMNS, daily_columns
from castcheck.store import TRUTH_COLUMNS
from castcheck.verify import (
    ALL_STATIONS,
    DEBIAS_MIN_HISTORY,
    MIN_N_CI,
    PAIRWISE_COLUMNS,
    PERSISTENCE_ID,
    SCORE_COLUMNS,
    VARIABLES,
    holm_reject,
    persistence_daily,
    score,
    select_truth,
    wilson_interval,
)

STATIONS = ("KAAA", "KBBB")
DATES = pd.date_range("2026-06-01", "2026-08-29", freq="D")          # 90 scored days
LEAD_IN = pd.date_range("2026-05-25", "2026-05-31", freq="D")        # baseline history only
ALL_DATES = LEAD_IN.append(DATES)
#: (hour, offset from the day's base temperature); the true peak/trough fall between the samples.
INSTANTS = ((6, -4.0), (12, -1.0), (18, 4.0), (0, 1.0))
CLI_MAX_EXTRA = 2.0
CLI_MIN_EXTRA = 1.5
LEADS = (1, 3)

#: per-instant error added by each model (hour -> °C); `noisy` is handled separately
OFFSETS = {
    "exact": dict.fromkeys((6, 12, 18, 0), 0.0),
    "warm1": dict.fromkeys((6, 12, 18, 0), 1.0),
    "cold18": {6: 0.0, 12: 0.0, 18: -3.0, 0: 0.0},
}


def _base(station: str, i: int) -> float:
    return 20.0 + (3.0 if station == "KBBB" else 0.0) + 6.0 * np.sin(i / 9.0)


def _valid_time(day: pd.Timestamp, hour: int) -> pd.Timestamp:
    """The four common instants of a climatological day: 06/12/18 UTC of D and 00 UTC of D+1."""
    return pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1 if hour == 0 else 0, hours=hour)


def make_truth_instant() -> pd.DataFrame:
    rows = []
    for st in STATIONS:
        for i, day in enumerate(ALL_DATES):
            for hour, delta in INSTANTS:
                rows.append({
                    "station_id": st, "valid_time": _valid_time(day, hour),
                    "temp_c": _base(st, i) + delta, "obs_time": _valid_time(day, hour),
                    "source": "ASOS_IEM", "n_reports": 1, "qc_flag": "",
                    "schema_version": "0.3", "methodology_version": "0.3",
                })
    return pd.DataFrame(rows)


def make_truth() -> pd.DataFrame:
    """CLI daily extremes: 2 °C above / 1.5 °C below the extremes of the four sampled instants."""
    rows = []
    for st in STATIONS:
        for i, day in enumerate(ALL_DATES):
            vals = [_base(st, i) + d for _h, d in INSTANTS]
            rows.append({
                "station_id": st, "climo_date": day.date(), "source": "CLI",
                "tmax_f": None, "tmin_f": None,
                "tmax_c": float(max(vals) + CLI_MAX_EXTRA),
                "tmin_c": float(min(vals) - CLI_MIN_EXTRA),
                "issuance_time": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=10),
                "is_final": True, "revised": False, "revised_tmax_f": None,
                "revised_tmin_f": None, "qc_flag": "", "product_id": f"p{i}",
                "schema_version": "0.3", "methodology_version": "0.3",
            })
    return pd.DataFrame(rows)[TRUTH_COLUMNS]


def make_frames(noise_sd: float = 2.0, seed: int = 11, days=None):
    """``(daily, instant)`` for four models at two leads, one init hour, one method."""
    days = DATES if days is None else days
    day_set = set(pd.DatetimeIndex(days))
    rng = np.random.default_rng(seed)
    inst_rows, daily_rows = [], []
    for st in STATIONS:
        for model_id in (*OFFSETS, "noisy"):
            for lead in LEADS:
                for i, day in enumerate(ALL_DATES):
                    if day not in day_set:
                        continue
                    init = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=lead)
                    fc, ob = [], []
                    day_noise = rng.normal(0, noise_sd)
                    for hour, delta in INSTANTS:
                        obs = _base(st, i) + delta
                        err = (day_noise if model_id == "noisy" else OFFSETS[model_id][hour])
                        valid = _valid_time(day, hour)
                        inst_rows.append({
                            "model_id": model_id, "model_version": "v1", "init_time": init,
                            "station_id": st, "valid_time": valid, "method": "bilinear",
                            "fcst_c": obs + err, "obs_c": obs, "err_c": err,
                            "lead_h": int((valid - init).total_seconds() // 3600),
                            "valid_hour_utc": hour, "lead_day": lead,
                            "climo_date": day.date(), "qc_flag": "",
                        })
                        fc.append(obs + err)
                        ob.append(obs)
                    daily_rows.append({
                        "model_id": model_id, "model_version": "v1", "init_time": init,
                        "station_id": st, "climo_date": day.date(), "lead_day": lead,
                        "method": "bilinear",
                        "tmax_sampled_c": max(fc), "tmin_sampled_c": min(fc), "n_samples": 4,
                        "tmax_native_c": max(fc) + CLI_MAX_EXTRA,
                        "tmin_native_c": min(fc) - CLI_MIN_EXTRA,
                        "missing_reason": "",
                        "tmax_obs_s_c": max(ob), "tmin_obs_s_c": min(ob), "n_obs_samples": 4,
                        "native_overhang_h": 0.0,
                        "schema_version": "0.3", "methodology_version": "0.3",
                    })
    daily = pd.DataFrame(daily_rows).reindex(columns=daily_columns())
    instant = pd.DataFrame(inst_rows)[INSTANT_ERROR_COLUMNS]
    return daily, instant


@pytest.fixture(scope="module")
def truth():
    return make_truth()


@pytest.fixture(scope="module")
def truth_instant():
    return make_truth_instant()


@pytest.fixture(scope="module")
def frames():
    return make_frames()


@pytest.fixture(scope="module")
def scored(frames, truth, truth_instant):
    daily, instant = frames
    return score(daily, truth, instant, windows=(30, 90, None), n_boot=400, seed=0,
                 truth_instant=truth_instant)


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
    sel2 = select_truth(t[t["source"] != "CLI"])
    assert pick(sel2, variable="tmax")["obs_c"] == 11.0
    sel3 = select_truth(t[t["source"] == "OBS"])
    assert pick(sel3, variable="tmax")["obs_c"] == 10.0
    assert pick(sel3, variable="tmax")["qc_flag"] == "obs_fallback"


# --------------------------------------------------------------------------------- schema

def test_score_schema(scored):
    scores, pairwise = scored
    assert list(scores.columns) == SCORE_COLUMNS
    assert list(pairwise.columns) == PAIRWISE_COLUMNS
    assert set(scores["window"]) == {"30d", "90d", "all"}
    assert set(scores["variable"]) == set(VARIABLES)
    assert PERSISTENCE_ID in set(scores["model_id"])


# ------------------------------------------------------------------- the headline variables

def test_t2_is_the_instantaneous_error_pooled_over_the_four_instants(scored):
    scores, _ = scored
    r = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="t2",
             method="bilinear", init_hour=0, window="90d")
    assert r["n"] == len(DATES)
    assert r["mae"] == pytest.approx(1.0, abs=1e-6)
    assert r["bias"] == pytest.approx(1.0, abs=1e-6)
    exact = pick(scores, station_id="KAAA", model_id="exact", lead_day=1, variable="t2",
                 method="bilinear", init_hour=0, window="90d")
    assert exact["mae"] == pytest.approx(0.0, abs=1e-9)
    # cold18 is 3 °C cold at one of the four instants: pooled |e| = 3/4
    cold = pick(scores, station_id="KAAA", model_id="cold18", lead_day=1, variable="t2",
                method="bilinear", init_hour=0, window="90d")
    assert cold["mae"] == pytest.approx(0.75, abs=1e-6)
    assert cold["bias"] == pytest.approx(-0.75, abs=1e-6)


def test_per_instant_variables_isolate_the_hour(scored):
    scores, _ = scored
    for variable, expect in (("t2_00z", 0.0), ("t2_06z", 0.0), ("t2_12z", 0.0), ("t2_18z", -3.0)):
        r = pick(scores, station_id="KAAA", model_id="cold18", lead_day=1, variable=variable,
                 method="bilinear", init_hour=0, window="90d")
        assert r["bias"] == pytest.approx(expect, abs=1e-6), variable
        assert r["n"] == len(DATES)


def test_sampled_extremes_are_like_for_like_and_cli_carries_the_sampling_penalty(scored):
    """Review item A2: a perfect model must score 0 on `tmax_s` and −2 °C on `tmax_cli`."""
    scores, _ = scored
    for variable, expect in (("tmax_s", 0.0), ("tmin_s", 0.0),
                             ("tmax_cli", -CLI_MAX_EXTRA), ("tmin_cli", CLI_MIN_EXTRA)):
        r = pick(scores, station_id="KAAA", model_id="exact", lead_day=1, variable=variable,
                 method="bilinear", init_hour=0, window="90d")
        assert r["bias"] == pytest.approx(expect, abs=1e-5), variable
        assert r["mae"] == pytest.approx(abs(expect), abs=1e-5), variable
    # the native extreme reproduces the CLI extreme exactly in this fixture
    nat = pick(scores, station_id="KAAA", model_id="exact", lead_day=1, variable="tmax_native_cli",
               method="bilinear", init_hour=0, window="90d")
    assert nat["bias"] == pytest.approx(0.0, abs=1e-5)


def test_window_lengths(scored):
    scores, _ = scored
    r30 = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="t2",
               method="bilinear", init_hour=0, window="30d")
    assert r30["n"] == 30
    assert str(r30["period_start"]) == "2026-07-31"
    r_all = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="t2",
                 method="bilinear", init_hour=0, window="all")
    assert r_all["n"] == len(DATES)


# ---------------------------------------------------------------------------- persistence

def test_persistence_baseline_is_the_same_functional_lead_days_earlier(scored, truth_instant):
    scores, _ = scored
    ti = truth_instant.set_index(["station_id", "valid_time"])["temp_c"]
    r = pick(scores, station_id="KAAA", model_id=PERSISTENCE_ID, lead_day=3, variable="t2",
             method="bilinear", init_hour=0, window="90d")
    expected = np.mean([
        np.mean([
            abs(ti[("KAAA", _valid_time(d, h))]
                - ti[("KAAA", _valid_time(d - pd.Timedelta(days=3), h))])
            for h, _ in INSTANTS
        ])
        for d in DATES
    ])
    assert r["mae"] == pytest.approx(expected, abs=1e-6)
    # tmax_s: the *observed sampled* extreme three days earlier, not the CLI extreme.  In this
    # fixture the CLI extreme is the sampled one plus a constant, so on a window where both
    # baselines see the same days their day-by-day errors — and therefore their bias — coincide;
    # neither carries a sampling offset, which is the point (the model side does).
    rs = pick(scores, station_id="KAAA", model_id=PERSISTENCE_ID, lead_day=3, variable="tmax_s",
              method="bilinear", init_hour=0, window="30d")
    rc = pick(scores, station_id="KAAA", model_id=PERSISTENCE_ID, lead_day=3, variable="tmax_cli",
              method="bilinear", init_hour=0, window="30d")
    assert rs["n"] == rc["n"] == 30
    assert rs["bias"] == pytest.approx(rc["bias"], abs=1e-6)
    assert rs["mae"] == pytest.approx(rc["mae"], abs=1e-6)


def test_skill_is_reproducible_from_the_published_denominator(scored):
    """Review item A1: `skill_persistence` must be recomputable from the row itself."""
    scores, _ = scored
    for model_id in ("exact", "warm1", "cold18", "noisy"):
        r = pick(scores, station_id="KAAA", model_id=model_id, lead_day=1, variable="t2",
                 method="bilinear", init_hour=0, window="90d")
        assert r["n_common"] == r["n"]           # full overlap in this fixture
        assert r["mae_persistence_common"] > 0
        assert r["skill_persistence"] == pytest.approx(
            1.0 - r["mae"] / r["mae_persistence_common"], abs=1e-6
        )
    pers = pick(scores, station_id="KAAA", model_id=PERSISTENCE_ID, lead_day=1, variable="t2",
                method="bilinear", init_hour=0, window="90d")
    assert np.isnan(pers["skill_persistence"])   # the baseline has no skill against itself
    assert pers["n_common"] == 0


def test_like_for_like_baseline_makes_a_perfect_forecast_score_full_skill(scored):
    """Review item B3: with an observed-extreme baseline the skill is a forecast statement again."""
    scores, _ = scored
    r = pick(scores, station_id="KAAA", model_id="exact", lead_day=1, variable="tmax_s",
             method="bilinear", init_hour=0, window="90d")
    assert r["skill_persistence"] == pytest.approx(1.0, abs=1e-6)
    assert r["skill_ci_low"] <= 1.0 <= r["skill_ci_high"] + 1e-9


# ------------------------------------------------------------------ out-of-sample debiasing

def test_debiased_mae_is_out_of_sample(scored):
    """Review item B1: the bias must come from days *before* the day being scored."""
    scores, _ = scored
    r = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="t2",
             method="bilinear", init_hour=0, window="all")
    # 90 scored days, the first 15 have too little history
    assert r["n_debiased"] == len(DATES) - DEBIAS_MIN_HISTORY
    # a constant +1 °C offset is estimated exactly from the history, so nothing is left over
    assert r["mae_debiased"] == pytest.approx(0.0, abs=1e-6)
    assert r["mae_debiased"] < r["mae"]
    noisy = pick(scores, station_id="KAAA", model_id="noisy", lead_day=1, variable="t2",
                 method="bilinear", init_hour=0, window="all")
    # out of sample the debiased error of a zero-mean noisy series is not smaller than the raw one:
    # the bias estimate itself has variance, which is exactly what the in-sample version hid
    assert noisy["mae_debiased"] > 0.9 * noisy["mae"]


# ---------------------------------------------------------------------------- uncertainty

def test_bootstrap_interval_brackets_the_truth_and_is_reproducible(frames, truth, truth_instant):
    daily, instant = frames
    kw = {"windows": (None,), "n_boot": 600, "truth_instant": truth_instant}
    scores, _ = score(daily, truth, instant, seed=0, **kw)
    r = pick(scores, station_id="KAAA", model_id="noisy", lead_day=1, variable="t2",
             method="bilinear", init_hour=0, window="all")
    population_mae = 2.0 * np.sqrt(2.0 / np.pi)
    assert r["mae_ci_low"] < population_mae < r["mae_ci_high"]
    assert r["bias_ci_low"] < 0.0 < r["bias_ci_high"]
    assert r["mae_ci_low"] < r["mae"] < r["mae_ci_high"]
    again, _ = score(daily, truth, instant, seed=0, **kw)
    r2 = pick(again, station_id="KAAA", model_id="noisy", lead_day=1, variable="t2",
              method="bilinear", init_hour=0, window="all")
    assert r2["mae_ci_low"] == r["mae_ci_low"] and r2["mae_ci_high"] == r["mae_ci_high"]
    other, _ = score(daily, truth, instant, seed=1, **kw)
    r3 = pick(other, station_id="KAAA", model_id="noisy", lead_day=1, variable="t2",
              method="bilinear", init_hour=0, window="all")
    assert r3["mae_ci_low"] != r["mae_ci_low"]


def test_identical_day_samples_give_identical_intervals_in_every_window(truth, truth_instant):
    """Regression for review item A3.

    A group whose realised days all sit inside the shortest window has *the same* sample in every
    window.  v0.2 drew the resample on the full window axis and produced four different intervals
    for one point estimate; v0.3 draws it on the group's own dates, so the intervals must be equal.
    """
    days = DATES[-MIN_N_CI:]
    daily, instant = make_frames(days=days)
    scores, pairwise = score(daily, truth, instant, windows=(30, 90, 365, None),
                             n_boot=300, seed=0, truth_instant=truth_instant)
    sub = scores[(scores["station_id"] == "KAAA") & (scores["model_id"] == "noisy")
                 & (scores["lead_day"] == 1) & (scores["variable"] == "t2")
                 & (scores["method"] == "bilinear") & (scores["init_hour"] == 0)]
    assert len(sub) == 4
    assert sub["n"].nunique() == 1
    assert sub["mae"].round(12).nunique() == 1
    for col in ("mae_ci_low", "mae_ci_high", "bias_ci_low", "bias_ci_high",
                "rmse_ci_low", "rmse_ci_high", "hit1f_ci_low", "hit1f_ci_high"):
        assert sub[col].round(12).nunique() == 1, col
    pw = pairwise[(pairwise["station_id"] == "KAAA") & (pairwise["lead_day"] == 1)
                  & (pairwise["variable"] == "t2") & (pairwise["model_a"] == "cold18")
                  & (pairwise["model_b"] == "noisy")]
    assert len(pw) == 4
    assert pw["ci_low"].round(12).nunique() == 1
    assert pw["p_boot"].round(12).nunique() == 1


def test_no_interval_below_the_minimum_sample(truth, truth_instant):
    days = DATES[-(MIN_N_CI - 1):]
    daily, instant = make_frames(days=days)
    scores, _ = score(daily, truth, instant, windows=(None,), n_boot=200, seed=0,
                      truth_instant=truth_instant)
    r = pick(scores, station_id="KAAA", model_id="noisy", lead_day=1, variable="t2",
             method="bilinear", init_hour=0, window="all")
    assert r["n"] == MIN_N_CI - 1
    assert np.isnan(r["mae_ci_low"]) and np.isnan(r["mae_ci_high"])
    assert np.isnan(r["skill_ci_low"])
    assert r["mae"] > 0                      # the point estimate is still published
    assert np.isfinite(r["hit1f_ci_low"])    # Wilson does not need the bootstrap


def test_hit_rate_interval_is_wilson_and_never_degenerate(scored):
    scores, _ = scored
    r = pick(scores, station_id="KAAA", model_id="cold18", lead_day=1, variable="t2_18z",
             method="bilinear", init_hour=0, window="90d")
    assert r["hit1f"] == 0.0                       # 3 °C is far outside ±1 °F
    assert r["hit1f_ci_low"] == 0.0
    assert 0.0 < r["hit1f_ci_high"] < 0.1          # a rule-of-three-like upper bound, not 0
    lo, hi = wilson_interval(0.0, r["n"])
    assert (float(lo), float(hi)) == pytest.approx((r["hit1f_ci_low"], r["hit1f_ci_high"]))


def test_wilson_matches_the_closed_form():
    lo, hi = wilson_interval(np.array([0.0, 5.0, 20.0]), np.array([20.0, 20.0, 20.0]))
    assert lo[0] == 0.0 and hi[0] == pytest.approx(0.1613, abs=1e-3)
    assert (lo[1], hi[1]) == pytest.approx((0.1119, 0.4687), abs=1e-3)
    assert lo[2] == pytest.approx(0.8389, abs=1e-3) and hi[2] == 1.0


def test_constant_error_gives_a_degenerate_interval(scored):
    scores, _ = scored
    r = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="t2",
             method="bilinear", init_hour=0, window="90d")
    assert r["mae_ci_low"] == pytest.approx(1.0, abs=1e-5)
    assert r["mae_ci_high"] == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------------------- ALL rows

def test_all_station_rows_average_the_daily_station_errors(scored):
    scores, _ = scored
    a = pick(scores, station_id=ALL_STATIONS, model_id="cold18", lead_day=1, variable="t2_18z",
             method="bilinear", init_hour=0, window="90d")
    assert a["mae"] == pytest.approx(3.0, abs=1e-6)
    assert a["n"] == len(DATES)
    assert a["n_stations"] == pytest.approx(2.0)
    assert set(scores.loc[scores["station_id"] == ALL_STATIONS, "model_id"]) == \
        set(scores.loc[scores["station_id"] == "KAAA", "model_id"])


# -------------------------------------------------------------------------------- pairwise

def test_pairwise_detects_a_clear_difference(scored):
    _, pairwise = scored
    row = pairwise[(pairwise["station_id"] == "KAAA") & (pairwise["lead_day"] == 1)
                   & (pairwise["variable"] == "t2") & (pairwise["window"] == "90d")
                   & (pairwise["method"] == "bilinear")
                   & (pairwise["model_a"] == "cold18") & (pairwise["model_b"] == "warm1")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["n_common"] == len(DATES)
    assert r["mae_diff"] == pytest.approx(-0.25, abs=1e-6)  # 0.75 − 1.0
    assert r["ci_high"] < 0
    assert bool(r["distinguishable_uncorrected"]) is True
    assert r["p_boot"] < 0.05


def test_pairwise_is_not_significant_when_models_are_identical(truth, truth_instant, frames):
    daily, instant = frames
    dtwin = daily[daily["model_id"] == "noisy"].copy()
    itwin = instant[instant["model_id"] == "noisy"].copy()
    dtwin["model_id"] = "noisy_twin"
    itwin["model_id"] = "noisy_twin"
    _, pairwise = score(pd.concat([daily, dtwin], ignore_index=True), truth,
                        pd.concat([instant, itwin], ignore_index=True),
                        windows=(90,), n_boot=400, seed=0, truth_instant=truth_instant)
    row = pairwise[(pairwise["station_id"] == "KAAA") & (pairwise["lead_day"] == 1)
                   & (pairwise["variable"] == "t2")
                   & (pairwise["model_a"] == "noisy") & (pairwise["model_b"] == "noisy_twin")]
    assert len(row) == 1
    assert row.iloc[0]["mae_diff"] == pytest.approx(0.0, abs=1e-9)
    assert bool(row.iloc[0]["distinguishable_uncorrected"]) is False
    assert bool(row.iloc[0]["distinguishable_holm"]) is False
    assert row.iloc[0]["p_boot"] == pytest.approx(1.0)


def test_pairwise_uses_only_common_days(truth, truth_instant, frames):
    daily, instant = frames
    keep_from = pd.Timestamp("2026-08-01")
    dshort = daily[(daily["model_id"] == "warm1")
                   & (pd.to_datetime(daily["climo_date"]) >= keep_from)].copy()
    ishort = instant[(instant["model_id"] == "warm1")
                     & (pd.to_datetime(instant["climo_date"]) >= keep_from)].copy()
    dshort["model_id"] = "warm1_short"
    ishort["model_id"] = "warm1_short"
    _, pairwise = score(pd.concat([daily, dshort], ignore_index=True), truth,
                        pd.concat([instant, ishort], ignore_index=True),
                        windows=(None,), n_boot=200, seed=0, truth_instant=truth_instant)
    row = pairwise[(pairwise["station_id"] == "KAAA") & (pairwise["lead_day"] == 1)
                   & (pairwise["variable"] == "t2")
                   & (pairwise["model_a"] == "cold18") & (pairwise["model_b"] == "warm1_short")]
    assert len(row) == 1
    assert row.iloc[0]["n_common"] == 29  # 2026-08-01 .. 2026-08-29


def test_holm_is_stricter_than_the_uncorrected_flag(scored):
    _, pairwise = scored
    fam = pairwise[pairwise["distinguishable_holm"]]
    assert len(fam)
    # Holm never marks a pair the single-comparison rule would not have marked
    assert bool(fam["distinguishable_uncorrected"].all())
    assert pairwise["distinguishable_holm"].sum() < pairwise["distinguishable_uncorrected"].sum()


def test_holm_reject_matches_the_textbook_step_down():
    p = np.array([0.001, 0.02, 0.04, 0.3])
    # m=4: 0.001·4 = .004 ✓ ; 0.02·3 = .06 ✗ → everything after is retained
    assert list(holm_reject(p)) == [True, False, False, False]
    assert list(holm_reject(np.array([0.001, 0.002, 0.003]))) == [True, True, True]
    assert list(holm_reject(np.array([]))) == []


# --------------------------------------------------------------------------------- corners

def test_small_samples_are_scored_but_reported_with_their_n(truth, truth_instant):
    days = DATES[-10:]
    daily, instant = make_frames(days=days)
    scores, _ = score(daily, truth, instant, windows=(None,), n_boot=200, seed=0,
                      truth_instant=truth_instant)
    r = pick(scores, station_id="KAAA", model_id="warm1", lead_day=1, variable="t2",
             method="bilinear", init_hour=0, window="all")
    assert r["n"] == 10
    assert r["mae"] == pytest.approx(1.0, abs=1e-6)
    assert r["n_debiased"] == 0            # never enough history in a 10-day series
    assert np.isnan(r["mae_debiased"])


def test_scores_without_the_instant_table_cover_the_daily_variables_only(frames, truth):
    daily, _ = frames
    scores, _ = score(daily, truth, None, windows=(None,), n_boot=0)
    assert set(scores["variable"]) == {"tmax_s", "tmin_s", "tmax_cli", "tmin_cli",
                                       "tmax_native_cli", "tmin_native_cli"}
    assert list(scores.columns) == SCORE_COLUMNS


def test_empty_inputs_return_empty_tables():
    empty_daily = pd.DataFrame(columns=daily_columns())
    empty_truth = pd.DataFrame(columns=TRUTH_COLUMNS)
    scores, pairwise = score(empty_daily, empty_truth)
    assert scores.empty and pairwise.empty
    assert list(scores.columns) == SCORE_COLUMNS
    assert list(pairwise.columns) == PAIRWISE_COLUMNS


def test_legacy_persistence_daily_still_produces_the_daily_schema(truth):
    pers = persistence_daily(truth)
    assert list(pers.columns) == daily_columns()
    assert set(pers["model_id"]) == {PERSISTENCE_ID}
    assert 0 not in set(pers["lead_day"])
    obs = {(r.station_id, r.climo_date): r.tmax_c for r in truth.itertuples()}
    sub = pers[(pers["station_id"] == "KAAA") & (pers["lead_day"] == 3)
               & (pers["method"] == "bilinear") & (pers["init_time"].dt.hour == 0)]
    for _, r in sub.head(10).iterrows():
        source_day = (pd.Timestamp(r["climo_date"]) - pd.Timedelta(days=3)).date()
        assert r["tmax_sampled_c"] == pytest.approx(obs[("KAAA", source_day)], abs=1e-5)
