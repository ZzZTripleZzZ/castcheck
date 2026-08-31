"""Independent check of the bootstrap used in :mod:`castcheck.verify` (METHODOLOGY §5).

Four questions, answered on the real archive in ``data/``:

1. Does the published interval agree with a textbook per-group moving-block percentile bootstrap
   computed from scratch on the same days?  (v0.3 removed the shared-resample-matrix approximation,
   so the only difference left should be Monte-Carlo noise.)
2. **Review item A3:** does a group whose realised days are identical in two windows get the same
   interval in both?  v0.2 did not; v0.3 must.
3. Does the paired difference interval really use only the days both models have?
4. How autocorrelated are the daily errors, and how much wider is a moving-block bootstrap than an
   i.i.d. one?

Run with ``PYTHONPATH=. .venv/bin/python scripts/crosscheck_bootstrap.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from castcheck import store
from castcheck.derive import instant_errors
from castcheck.verify import BLOCK_DAYS, MIN_N_CI, error_table

N_BOOT = 4000
SEED = 12345
KEY = ["station_id", "model_id", "init_hour", "lead_day", "variable", "method"]


def standard_bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED,
                          block: int = 1) -> tuple[float, float]:
    """Textbook percentile bootstrap of the mean of ``x``; ``block>1`` = circular moving block."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return float(x.mean()), float(x.mean())
    if block <= 1:
        idx = rng.integers(0, n, size=(n_boot, n))
    else:
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(n_boot, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
        idx = idx.reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def acf(x: np.ndarray, lags: int = 7) -> list[float]:
    x = x - x.mean()
    denom = float((x * x).sum())
    return [float((x[:-k] * x[k:]).sum() / denom) if denom else np.nan for k in range(1, lags + 1)]


def _unit_series(err: pd.DataFrame, key: tuple) -> pd.DataFrame:
    """One row per scored day for one group: the mean of |e| over the day's instants."""
    g = err[(err[KEY] == pd.Series(dict(zip(KEY, key)))).all(axis=1)]
    per = (g.assign(a=g["err"].abs())
             .groupby("climo_date", observed=True)["a"].mean()
             .reset_index().sort_values("climo_date"))
    return per


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    daily = store.read_daily()
    truth = store.read_truth()
    ti = store.read_truth_instant()
    instant = None
    if len(ti):
        from castcheck.derive import DERIVE_VALUE_COLUMNS

        values = store.read_forecast_values(columns=DERIVE_VALUE_COLUMNS)
        instant = instant_errors(values, ti)
    err = error_table(daily, truth, instant)
    err["climo_date"] = pd.to_datetime(err["climo_date"])
    return daily, truth, ti, err


def main() -> None:
    daily, truth, ti, err = load()
    if err.empty:
        print("no scored rows in data/ — run `castcheck derive && castcheck verify` first")
        return

    sizes = err.groupby(KEY, observed=True)["climo_date"].nunique().sort_values(ascending=False)
    picks = list(sizes[sizes >= MIN_N_CI].index[:6]) + list(sizes[sizes < MIN_N_CI].index[:2])

    # the comparison uses the *published* table, so that it checks what the site actually shows
    published, published_pw = store.read_scores()
    if published.empty:
        print("no published scores — run `castcheck verify` first")
        return

    print(f"{'group':<66} {'n':>4} | textbook block CI  | published CI       | width ratio")
    print("-" * 122)
    ratios = []
    for k in picks:
        per = _unit_series(err, k)
        a = per["a"].to_numpy()
        s_lo, s_hi = standard_bootstrap_ci(a, block=BLOCK_DAYS)
        row = published[(published[KEY] == pd.Series(dict(zip(KEY, k)))).all(axis=1)
                        & (published["window"] == "all")]
        label = "/".join(str(x) for x in k)
        if row.empty:
            print(f"{label:<66} {len(a):>4} | [{s_lo:6.3f}, {s_hi:6.3f}] | (not published)")
            continue
        r0 = row.iloc[0]
        p_lo, p_hi = float(r0["mae_ci_low"]), float(r0["mae_ci_high"])
        if not np.isfinite(p_lo):
            print(f"{label:<66} {len(a):>4} | [{s_lo:6.3f}, {s_hi:6.3f}] |"
                  f" — (n<{MIN_N_CI}, no CI by design)")
            continue
        ratio = (p_hi - p_lo) / (s_hi - s_lo) if s_hi > s_lo else np.nan
        ratios.append(ratio)
        print(f"{label:<66} {len(a):>4} | [{s_lo:6.3f}, {s_hi:6.3f}] |"
              f" [{p_lo:6.3f}, {p_hi:6.3f}] | {ratio:5.3f}")
    if ratios:
        print(f"\nmedian width ratio (published / textbook per-group) = {np.nanmedian(ratios):.4f}"
              "  — 1.00 ± Monte-Carlo noise is the expectation\n")

    # --- A3: the same realised days must give the same interval in every window ---------------
    print("A3 regression — one point estimate, four windows:")
    cols = ["window", "n", "mae", "mae_ci_low", "mae_ci_high"]
    same = (published.groupby(KEY, observed=True)["n"].nunique() == 1)
    cand = published.merge(same.rename("same_n").reset_index(), on=KEY)
    cand = cand[cand["same_n"] & (cand["n"] >= MIN_N_CI)]
    if cand.empty:
        print("  no group with an identical sample in every window (needs a sparser archive)")
    else:
        k = cand.iloc[0][KEY].to_dict()
        sub = published[(published[KEY] == pd.Series(k)).all(axis=1)].sort_values("window")
        print("  " + "/".join(str(v) for v in k.values()))
        print(sub[cols].to_string(index=False))
        widths = (sub["mae_ci_high"] - sub["mae_ci_low"]).round(12).nunique()
        print(f"  distinct interval widths across windows: {widths}"
              f"  ({'OK' if widths == 1 else 'MISMATCH — A3 not fixed'})")

    # --- autocorrelation and block width ------------------------------------------------------
    print(f"\n{'group':<66} {'n':>4} | acf lag1..5                       | iid width | block7 | ratio")
    print("-" * 130)
    for k in picks:
        per = _unit_series(err, k)
        if len(per) < 25:
            continue
        a = per["a"].to_numpy()
        r = acf(a, 5)
        i_lo, i_hi = standard_bootstrap_ci(a, block=1)
        b_lo, b_hi = standard_bootstrap_ci(a, block=BLOCK_DAYS)
        label = "/".join(str(x) for x in k)
        print(f"{label:<66} {len(a):>4} | {' '.join(f'{v:6.3f}' for v in r)} |"
              f" {i_hi - i_lo:9.3f} | {b_hi - b_lo:6.3f} | {(b_hi - b_lo) / (i_hi - i_lo):5.3f}")

    # --- paired differences use only the common days -------------------------------------------
    if len(published_pw):
        pw = published_pw[published_pw["window"] == "all"].head(3)
        print("\npaired comparisons — n_common vs each side's own n:")
        for _, r in pw.iterrows():
            look = {c: r[c] for c in ("station_id", "init_hour", "lead_day", "variable", "method")}
            na = published[(published[list(look)] == pd.Series(look)).all(axis=1)
                           & (published["model_id"] == r["model_a"])
                           & (published["window"] == "all")]["n"]
            nb = published[(published[list(look)] == pd.Series(look)).all(axis=1)
                           & (published["model_id"] == r["model_b"])
                           & (published["window"] == "all")]["n"]
            na = int(na.iloc[0]) if len(na) else -1
            nb = int(nb.iloc[0]) if len(nb) else -1
            ok = r["n_common"] <= min(na, nb)
            print(f"  {r['model_a']:>16} vs {r['model_b']:<16} n_a={na:4d} n_b={nb:4d}"
                  f" n_common={int(r['n_common']):4d}  {'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
