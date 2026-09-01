"""Tests for the monthly incremental-vs-full self-check (`crosscheck_verify.py --compare-incremental`).

Part 1 and part 2 of that script are exercised by running it against the real archive; what is worth
unit-testing is part 3, because it is the thing that decides whether `consistency.yml` files an issue
saying the published numbers are wrong. A comparator that is too lax never fires, and one that is too
strict fires every month until people stop reading it — so the tests pin both edges: a float that
differs in the last bits must pass, a drift of 1e-5 must fail, and a row that exists on one side only
must fail rather than be quietly skipped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "crosscheck_verify", REPO_ROOT / "scripts" / "crosscheck_verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load()


def _scores(rows: list[dict]) -> pd.DataFrame:
    base = {"station_id": "ALL", "model_id": "gfs", "init_hour": 0, "lead_day": 1,
            "variable": "t2", "method": "bilinear", "window": "90d",
            "n": 90.0, "n_debiased": 60.0, "n_common": 88.0, "mae": 1.5, "bias": -0.2,
            "rmse": 1.9, "hit1f": 0.31, "hit2f": 0.58, "hit3f": 0.77,
            "mae_debiased": 1.4, "mae_persistence_common": 3.1, "skill_persistence": 0.52}
    return pd.DataFrame([{**base, **r} for r in rows])


@pytest.fixture
def compare(tmp_path, monkeypatch, capsys):
    """Run part3(full, incremental) and give back (exit_code, stdout)."""
    from castcheck import store

    def _run(full: pd.DataFrame, incremental: pd.DataFrame) -> tuple[int, str]:
        path = tmp_path / "incremental.parquet"
        incremental.to_parquet(path)
        monkeypatch.setattr(store, "read_scores", lambda: (full, pd.DataFrame()))
        rc = cc.part3(path)
        return rc, capsys.readouterr().out

    return _run


# --------------------------------------------------------------------------- the tolerance


def test_identical_tables_agree(compare):
    rows = _scores([{}, {"model_id": "ifs_hres", "mae": 1.2}, {"lead_day": 3, "mae": 2.4}])
    rc, out = compare(rows, rows.copy())
    assert rc == 0, out
    assert "OK: 0 difference" in out


def test_a_bit_level_float_difference_still_agrees(compare):
    """Parquet round-trips and a different summation order move the last bits; that is not a bug."""
    full = _scores([{"mae": 1.5}])
    inc = _scores([{"mae": 1.5 + 1e-12}])
    rc, out = compare(full, inc)
    assert rc == 0, out
    assert "OK: 0 difference" in out


def test_a_drift_of_1e_5_fails(compare):
    """1e-5 °C is invisible on the site and still means the two paths disagree — the point of the check."""
    rc, out = compare(_scores([{"mae": 1.5}]), _scores([{"mae": 1.50001}]))
    assert rc == 1
    assert "FAILED: 1 difference" in out
    assert "mae" in out


def test_the_tolerance_is_relative_above_unit_magnitude(compare):
    """`tol * max(1, |a|)`, the same rule part 1 uses: 1e-6 absolute on a value of 5000 would flag
    every parquet round-trip, while 1e-6 *relative* still catches any difference worth a look."""
    rc, _ = compare(_scores([{"n": 5000.0}]), _scores([{"n": 5000.0 + 1e-3}]))
    assert rc == 0                        # 2e-7 relative — inside 1e-6
    rc, _ = compare(_scores([{"n": 5000.0}]), _scores([{"n": 5000.0 + 0.1}]))
    assert rc == 1                        # 2e-5 relative — outside


# --------------------------------------------------------------------------- NaN and row sets


def test_two_nans_are_equal_but_a_nan_against_a_number_is_not():
    nan = float("nan")
    assert cc._close(nan, nan)
    assert not cc._close(nan, 1.0)
    assert not cc._close(1.0, nan)


def test_a_ci_that_became_nan_is_reported(compare):
    """`mae_debiased` is NaN for a group without enough history; gaining or losing one is a change."""
    rc, out = compare(_scores([{"mae_debiased": float("nan")}]), _scores([{"mae_debiased": 1.4}]))
    assert rc == 1
    assert "mae_debiased" in out


def test_a_row_only_the_full_recompute_produced_fails(compare):
    full = _scores([{}, {"model_id": "aurora_ifs"}])
    rc, out = compare(full, _scores([{}]))
    assert rc == 1
    assert "only in the full recompute" in out
    assert "aurora_ifs" in out


def test_a_row_that_disappeared_from_the_full_recompute_fails(compare):
    rc, out = compare(_scores([{}]), _scores([{}, {"model_id": "aurora_ifs"}]))
    assert rc == 1
    assert "only in the incremental scores" in out


def test_an_empty_published_table_is_an_error_not_an_all_clear(compare):
    rc, out = compare(pd.DataFrame(), _scores([{}]))
    assert rc == 1
    assert "no scores" in out
