"""Independent check of the bootstrap used in :mod:`castcheck.verify` (METHODOLOGY §5).

Three questions, answered on the real archive in ``data/``:

1. Is the shared-weight, self-normalised bootstrap that ``verify.score`` uses equivalent to a
   textbook per-group percentile bootstrap over days?
2. Does the paired difference interval really use only the days both models have?
3. How autocorrelated are the daily errors, and how much wider is a moving-block bootstrap?

Run with ``PYTHONPATH=. .venv/bin/python scripts/crosscheck_bootstrap.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from castcheck import store
from castcheck.verify import error_table, persistence_daily

N_BOOT = 4000
SEED = 12345


def standard_bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED,
                          block: int = 1) -> tuple[float, float]:
    """Textbook percentile bootstrap of the mean of ``x``; ``block>1`` = moving-block."""
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


def shared_w_ci(x: np.ndarray, present: np.ndarray, nd: int, n_boot: int = N_BOOT,
                seed: int = SEED) -> tuple[float, float]:
    """The verify.py approximation: one multinomial resample of *all* window days, shared by
    every group, evaluated as a self-normalised weighted mean over the group's own days."""
    rng = np.random.default_rng(seed)
    w = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=n_boot).astype("float64")
    full = np.zeros(nd)
    full[present] = x
    mask = np.zeros(nd)
    mask[present] = 1.0
    num = w @ full
    den = w @ mask
    den[den == 0] = np.nan
    lo, hi = np.nanpercentile(num / den, [2.5, 97.5])
    return float(lo), float(hi)


def acf(x: np.ndarray, lags: int = 7) -> list[float]:
    x = x - x.mean()
    denom = float((x * x).sum())
    return [float((x[:-k] * x[k:]).sum() / denom) if denom else np.nan for k in range(1, lags + 1)]


def main() -> None:
    daily = store.read_daily()
    truth = store.read_truth()
    daily = pd.concat([daily, persistence_daily(truth)], ignore_index=True)
    err = error_table(daily, truth)
    err["climo_date"] = pd.to_datetime(err["climo_date"])

    key = ["station_id", "model_id", "init_hour", "lead_day", "variable", "method"]
    sizes = err.groupby(key, observed=True).size().sort_values(ascending=False)
    picks = list(sizes[sizes >= 20].index[:6]) + list(sizes[(sizes >= 8) & (sizes < 20)].index[:4])

    all_dates = np.sort(err["climo_date"].unique())
    pos = pd.Series(np.arange(len(all_dates)), index=pd.DatetimeIndex(all_dates))

    print(f"{'group':<58} {'n':>4} | standard CI        | shared-W CI        | width ratio")
    print("-" * 118)
    ratios = []
    for k in picks:
        g = err[(err[key] == pd.Series(dict(zip(key, k)))).all(axis=1)].sort_values("climo_date")
        a = g["err"].abs().to_numpy()
        present = pos.reindex(pd.DatetimeIndex(g["climo_date"])).to_numpy()
        s_lo, s_hi = standard_bootstrap_ci(a)
        w_lo, w_hi = shared_w_ci(a, present, len(all_dates))
        r = (w_hi - w_lo) / (s_hi - s_lo) if s_hi > s_lo else np.nan
        ratios.append(r)
        label = "/".join(str(x) for x in k)
        print(f"{label:<58} {len(a):>4} | [{s_lo:6.3f}, {s_hi:6.3f}] | [{w_lo:6.3f}, {w_hi:6.3f}] | {r:5.3f}")
    print(f"\nmedian width ratio (shared-W / standard) = {np.nanmedian(ratios):.4f}\n")

    # --- autocorrelation and block bootstrap --------------------------------------------------
    print(f"{'group':<58} {'n':>4} | acf lag1..5                       | iid width | block5 | ratio")
    print("-" * 118)
    for k in picks:
        g = err[(err[key] == pd.Series(dict(zip(key, k)))).all(axis=1)].sort_values("climo_date")
        if len(g) < 25:
            continue
        a = g["err"].abs().to_numpy()
        r = acf(g["err"].to_numpy(), 5)
        i_lo, i_hi = standard_bootstrap_ci(a, block=1)
        b_lo, b_hi = standard_bootstrap_ci(a, block=5)
        label = "/".join(str(x) for x in k)
        print(f"{label:<58} {len(a):>4} | {' '.join(f'{v:6.3f}' for v in r)} |"
              f" {i_hi - i_lo:9.3f} | {b_hi - b_lo:6.3f} | {(b_hi - b_lo) / (i_hi - i_lo):5.3f}")


if __name__ == "__main__":
    main()
