"""Independent recomputation of published scores, in the most naive pandas possible.

Part 1 — pick a few ``(station, model, init_hour, lead_day, variable, method, window)`` groups from
``data/scores/latest.parquet`` and recompute ``n``, ``MAE``, ``bias``, ``RMSE`` and the hit rates
from ``daily_forecasts`` + ``truth_daily`` with plain ``for``-loop-free pandas that shares no code
with :mod:`castcheck.verify` beyond reading the tables.  Everything must agree to 1e-6.

Part 2 — compare one station-day of ``ifs_hres`` against Open-Meteo's Previous Runs API
(``previous-runs-api.open-meteo.com``, development-time sanity check only, non-commercial).  This is
a *magnitude* check on the extraction chain: Open-Meteo interpolates the same IFS HRES fields to the
same point but at 9 km native resolution with its own downscaling, so agreement is expected within
about ±1 °C, not to the last decimal.

Run with ``PYTHONPATH=. .venv/bin/python scripts/crosscheck_verify.py [--no-network]``.
"""

from __future__ import annotations

import json
import sys
import urllib.request

import numpy as np
import pandas as pd

from castcheck import store
from castcheck.config import station_by_id
from castcheck.verify import HIT_THRESHOLDS_C, PERSISTENCE_ID, persistence_daily

KEY = ["station_id", "model_id", "init_hour", "lead_day", "variable", "method", "window"]
TOL = 1e-6


def naive_error_rows(daily: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """One row per scored (group, day) built with the dumbest possible joins."""
    t = truth[truth["source"] == "CLI"].copy()
    t = t[t["is_final"].fillna(False).astype(bool)]
    t["climo_date"] = pd.to_datetime(t["climo_date"])
    obs = []
    for var in ("tmax", "tmin"):
        part = t[["station_id", "climo_date"]].copy()
        part["variable"] = var
        part["obs_c"] = pd.to_numeric(t[f"{var}_c"], errors="coerce")
        obs.append(part.dropna(subset=["obs_c"]))
    obs = pd.concat(obs, ignore_index=True)

    d = daily.copy()
    d["climo_date"] = pd.to_datetime(d["climo_date"])
    d["init_hour"] = pd.to_datetime(d["init_time"], utc=True).dt.hour
    fc = []
    for var in ("tmax", "tmin"):
        part = d[["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date"]].copy()
        part["variable"] = var
        part["fcst_c"] = pd.to_numeric(d[f"{var}_sampled_c"], errors="coerce")
        fc.append(part.dropna(subset=["fcst_c"]))
    fc = pd.concat(fc, ignore_index=True)

    m = fc.merge(obs, on=["station_id", "climo_date", "variable"], how="inner")
    m["err"] = m["fcst_c"] - m["obs_c"]
    return m


def window_slice(rows: pd.DataFrame, window: str, as_of: pd.Timestamp) -> pd.DataFrame:
    if window == "all":
        return rows
    days = int(window.rstrip("d"))
    return rows[rows["climo_date"] >= as_of - pd.Timedelta(days=days - 1)]


def part1() -> int:
    scores, _ = store.read_scores()
    daily = store.read_daily()
    truth = store.read_truth()
    if scores.empty or daily.empty:
        print("no scores/daily in data/ — run `castcheck derive && castcheck verify` first")
        return 1
    daily = pd.concat([daily, persistence_daily(truth)], ignore_index=True)
    rows = naive_error_rows(daily, truth)
    as_of = pd.to_datetime(truth["climo_date"]).max().normalize()

    # a real single-station model group, an ALL row, and a persistence row
    cand = scores[(scores["n"] >= 10)]
    picks = pd.concat([
        cand[(cand["station_id"] != "ALL") & (cand["model_id"] != PERSISTENCE_ID)].head(2),
        cand[cand["station_id"] == "ALL"].head(1),
        cand[cand["model_id"] == PERSISTENCE_ID].head(1),
    ])
    bad = 0
    print(f"{'group':<74} {'stat':<6} {'published':>12} {'naive':>12} {'Δ':>10}")
    print("-" * 120)
    for _, r in picks.iterrows():
        sub = rows[(rows["model_id"] == r["model_id"])
                   & (rows["init_hour"] == r["init_hour"])
                   & (rows["lead_day"] == r["lead_day"])
                   & (rows["variable"] == r["variable"])
                   & (rows["method"] == r["method"])]
        if r["station_id"] == "ALL":
            # METHODOLOGY §4: average across stations within a day, then over days
            per_day = sub.groupby("climo_date").agg(
                a=("err", lambda s: s.abs().mean()),
                s=("err", "mean"),
                q=("err", lambda s: (s ** 2).mean()),
                h1=("err", lambda s: (s.abs() <= HIT_THRESHOLDS_C[0] + 1e-9).mean()),
            ).reset_index()
        else:
            sub = sub[sub["station_id"] == r["station_id"]]
            per_day = pd.DataFrame({
                "climo_date": sub["climo_date"],
                "a": sub["err"].abs(), "s": sub["err"], "q": sub["err"] ** 2,
                "h1": (sub["err"].abs() <= HIT_THRESHOLDS_C[0] + 1e-9).astype(float),
            })
        per_day = window_slice(per_day, r["window"], as_of)
        got = {
            "n": float(len(per_day)),
            "mae": float(per_day["a"].mean()),
            "bias": float(per_day["s"].mean()),
            "rmse": float(np.sqrt(per_day["q"].mean())),
            "hit1f": float(per_day["h1"].mean()),
        }
        label = "/".join(str(r[k]) for k in KEY)
        for stat, val in got.items():
            pub = float(r[stat])
            delta = abs(pub - val)
            flag = "" if delta <= TOL * max(1.0, abs(pub)) else "  <-- MISMATCH"
            if flag:
                bad += 1
            print(f"{label:<74} {stat:<6} {pub:12.6f} {val:12.6f} {delta:10.2e}{flag}")
    print(f"\n{'FAILED' if bad else 'OK'}: {bad} mismatch(es) beyond {TOL:g}")
    return 1 if bad else 0


def part2() -> int:
    """Open-Meteo Previous Runs cross-check for one KNYC day of ifs_hres."""
    daily = store.read_daily(["ifs_hres"])
    if daily.empty:
        print("\nno ifs_hres daily rows — skipping the Open-Meteo check")
        return 0
    # compare the `nearest` variant: Open-Meteo returns the value of the containing grid cell, it
    # does not interpolate, so `nearest` is the like-for-like column.
    d = daily[(daily["station_id"] == "KNYC") & (daily["lead_day"] == 1)
              & (daily["method"] == "nearest")
              & (pd.to_datetime(daily["init_time"], utc=True).dt.hour == 0)]
    d = d.dropna(subset=["tmax_sampled_c"])
    if d.empty:
        print("\nno KNYC ifs_hres D+1 rows — skipping the Open-Meteo check")
        return 0
    row = d.sort_values("climo_date").iloc[-2 if len(d) > 1 else -1]
    st = station_by_id("KNYC")
    day = pd.Timestamp(row["climo_date"]).date()
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()
    url = (
        "https://previous-runs-api.open-meteo.com/v1/forecast"
        f"?latitude={st.lat}&longitude={st.lon}"
        "&hourly=temperature_2m_previous_day1&models=ecmwf_ifs025"
        f"&start_date={day}&end_date={end}"
        "&timezone=UTC&temperature_unit=celsius"
    )
    print(f"\nOpen-Meteo cross-check  station=KNYC  climo_date={day}  model=ifs_hres  lead_day=1")
    print(f"  {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host
            payload = json.load(resp)
    except Exception as exc:  # pragma: no cover - network
        print(f"  request failed ({exc}); skipping")
        return 0
    h = payload.get("hourly", {})
    ser = pd.Series(h.get("temperature_2m_previous_day1", []),
                    index=pd.to_datetime(h.get("time", []), utc=True)).dropna()
    if ser.empty:
        print("  no values returned; skipping")
        return 0
    # the same four common samples of the same LST day (METHODOLOGY §2.2)
    start = pd.Timestamp(day, tz="UTC") - pd.Timedelta(hours=st.std_offset_h)
    win = ser[(ser.index >= start) & (ser.index < start + pd.Timedelta(hours=24))]
    win = win[win.index.hour.isin((0, 6, 12, 18))]
    if len(win) < 4:
        print(f"  only {len(win)} of the 4 common samples returned; skipping")
        return 0
    om_max, om_min = float(win.max()), float(win.min())
    cc_max, cc_min = float(row["tmax_sampled_c"]), float(row["tmin_sampled_c"])
    print(f"  sampled Tmax  castcheck {cc_max:6.2f} °C   open-meteo {om_max:6.2f} °C"
          f"   Δ {cc_max - om_max:+.2f} °C")
    print(f"  sampled Tmin  castcheck {cc_min:6.2f} °C   open-meteo {om_min:6.2f} °C"
          f"   Δ {cc_min - om_min:+.2f} °C")
    ok = abs(cc_max - om_max) <= 1.5 and abs(cc_min - om_min) <= 1.5
    print(f"  {'OK' if ok else 'OUTSIDE ±1.5 °C — investigate'}: Open-Meteo's `previous_day1` is the"
          " run ~24 h before the valid time,\n  which is not exactly our 00 UTC initialization, so"
          " this is a magnitude check on the extraction chain,\n  not a bit-for-bit comparison.")
    return 0


if __name__ == "__main__":
    rc = part1()
    if "--no-network" not in sys.argv:
        part2()
    raise SystemExit(rc)
