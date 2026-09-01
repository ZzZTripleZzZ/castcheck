"""Independent recomputation of published scores, in the most naive pandas possible.

Part 1 — pick a few ``(station, model, init_hour, lead_day, variable, method, window)`` groups from
``data/scores/latest.parquet`` — deliberately one of each *kind* the v0.3 schema has: a pooled ``t2``
station row, a per-instant ``t2_*`` row, a like-for-like ``tmax_s`` row, an ``ALL`` row and a
``persistence`` row — and recompute ``n``, ``MAE``, ``bias``, ``RMSE`` and the ±1 °F hit rate from
``forecast_values`` + ``truth_instant`` + ``daily_forecasts`` + ``truth_daily`` with plain pandas
that shares no code with :mod:`castcheck.verify` beyond reading the tables.  Everything must agree
to 1e-6.

The independent implementation reproduces, from the methodology text alone:

* the four common instants of a climatological day (station standard offset, §2.2), and the rule
  that only a day with all four is scored;
* the pooled ``t2`` unit — a day's score is the mean over its four instants, not over station-days;
* the ``ALL`` row — the cross-station mean of each functional within a day, then the mean over days;
* the lagged-persistence baseline of the *same functional* (§4 / DESIGN §10.3).

Part 2 — compare one station-day of ``ifs_hres`` against Open-Meteo's Previous Runs API
(``previous-runs-api.open-meteo.com``, development-time sanity check only, non-commercial).  This is
a *magnitude* check on the extraction chain: Open-Meteo interpolates the same IFS HRES fields to the
same point but at 9 km native resolution with its own downscaling, so agreement is expected within
about ±1 °C, not to the last decimal.

Part 3 (``--compare-incremental OLD.parquet``) — the *incremental vs full* self-check run monthly by
``consistency.yml``.  The daily pipeline derives only the last 14 days of initialisations
(``derive --since 14``) and re-scores on top of whatever ``daily_forecasts`` already held; a bug in
the upsert path, a dtype that rounds, or a shard written by an older code version would show up as a
slow drift that no single day's run can see.  So once a month the whole archive is re-derived from
``forecast_values`` and re-scored, and every published statistic is required to match the incremental
answer to 1e-6.  ``as_of`` — and therefore the 30d/90d/365d window edges — is taken from the data
(``max(climo_date)``), not the clock, so the comparison is deterministic as long as both sides see
the same ``data/``; the workflow guarantees that by restoring ``data/`` to the commit that published
the incremental scores.

Run with ``PYTHONPATH=. .venv/bin/python scripts/crosscheck_verify.py [--no-network]`` or
``... scripts/crosscheck_verify.py --compare-incremental /tmp/incremental_scores.parquet``.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from castcheck import store
from castcheck.config import load_stations, station_by_id
from castcheck.verify import HIT_THRESHOLDS_C, PERSISTENCE_ID

KEY = ["station_id", "model_id", "init_hour", "lead_day", "variable", "method", "window"]
TOL = 1e-6
INSTANT_HOURS = (0, 6, 12, 18)
HOUR_VARIABLE = {h: f"t2_{h:02d}z" for h in INSTANT_HOURS}


# ------------------------------------------------------------------ naive scored-row builders

def _offsets() -> dict[str, int]:
    return {s.id: int(s.std_offset_h) for s in load_stations()}


def naive_instant_rows() -> pd.DataFrame:
    """`t2` and `t2_*` rows: the forecast at 00/06/12/18 UTC against the observation at that instant."""
    off = _offsets()
    v = store.read_forecast_values(
        columns=["model_id", "init_time", "valid_time", "station_id", "variable", "bucket_h",
                 "method", "value_c", "missing_reason"])
    v = v[(v["variable"] == "t2") & (v["bucket_h"] == 0) & (v["missing_reason"] == "")
          & v["value_c"].notna()]
    v = v.copy()
    v["init_time"] = pd.to_datetime(v["init_time"], utc=True)
    v["valid_time"] = pd.to_datetime(v["valid_time"], utc=True)
    v = v[v["valid_time"].dt.hour.isin(INSTANT_HOURS)]
    v["init_hour"] = v["init_time"].dt.hour
    v["_hour"] = v["valid_time"].dt.hour
    hours = pd.to_timedelta(v["station_id"].map(off).astype(int), unit="h")
    v["climo_date"] = (v["valid_time"] + hours).dt.floor("D").dt.tz_localize(None)
    v["lead_day"] = (v["climo_date"] - v["init_time"].dt.floor("D").dt.tz_localize(None)).dt.days
    grp = ["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date"]
    # a climatological day is scored only when the run covers all four of its instants (§2.5)
    v = v[v.groupby(grp, observed=True)["_hour"].transform("nunique") == 4]

    ti = store.read_truth_instant()
    ti = ti[ti["temp_c"].notna()][["station_id", "valid_time", "temp_c"]]
    m = v.merge(ti, on=["station_id", "valid_time"], how="inner")
    if m.empty:
        return m.assign(variable=[], err=[])
    m["err"] = m["value_c"] - m["temp_c"]
    # the *pooled* t2 additionally needs all four observations (§2.3); the per-hour variables do not
    pooled = m[m.groupby(grp, observed=True)["_hour"].transform("nunique") == 4].assign(variable="t2")
    hourly = m.assign(variable=m["_hour"].map(HOUR_VARIABLE))
    return pd.concat([pooled, hourly], ignore_index=True)[
        [*grp, "variable", "_hour", "valid_time", "init_time", "err", "temp_c"]]


def naive_daily_rows() -> pd.DataFrame:
    """`tmax_s/tmin_s` (vs the observed sampled extreme) and `*_cli` (vs the NWS CLI extreme)."""
    d = store.read_daily()
    d["init_time"] = pd.to_datetime(d["init_time"], utc=True)
    d["init_hour"] = d["init_time"].dt.hour
    d["climo_date"] = pd.to_datetime(d["climo_date"])
    t = store.read_truth()
    t = t[(t["source"] == "CLI") & t["is_final"].fillna(False).astype(bool)].copy()
    t["climo_date"] = pd.to_datetime(t["climo_date"])
    grp = ["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date"]
    out = []
    spec = [("tmax_s", "tmax_sampled_c", "tmax_obs_s_c", None),
            ("tmin_s", "tmin_sampled_c", "tmin_obs_s_c", None),
            ("tmax_cli", "tmax_sampled_c", None, "tmax_c"),
            ("tmin_cli", "tmin_sampled_c", None, "tmin_c"),
            ("tmax_native_cli", "tmax_native_c", None, "tmax_c"),
            ("tmin_native_cli", "tmin_native_c", None, "tmin_c")]
    for variable, fcol, ocol, tcol in spec:
        part = d[[*grp, "init_time"]].copy()
        part["fcst"] = pd.to_numeric(d[fcol], errors="coerce")
        if ocol is not None:
            part["obs"] = pd.to_numeric(d[ocol], errors="coerce")
        else:
            obs = t[["station_id", "climo_date"]].copy()
            obs["obs"] = pd.to_numeric(t[tcol], errors="coerce")
            part = part.merge(obs, on=["station_id", "climo_date"], how="inner")
        part = part.dropna(subset=["fcst", "obs"])
        part["variable"] = variable
        part["err"] = part["fcst"] - part["obs"]
        part["_hour"] = 0
        part["valid_time"] = pd.NaT
        out.append(part[[*grp, "variable", "_hour", "valid_time", "init_time", "err", "obs"]]
                   .rename(columns={"obs": "temp_c"}))
    return pd.concat(out, ignore_index=True)


def naive_persistence_rows(inst: pd.DataFrame, dly: pd.DataFrame) -> pd.DataFrame:
    """The observation `lead_day` days earlier — of the same functional (DESIGN §10.3)."""
    out = []
    leads = sorted({int(x) for x in pd.concat([inst["lead_day"], dly["lead_day"]]).unique()
                    if int(x) >= 1})
    init_hours = sorted(set(inst["init_hour"]) | set(dly["init_hour"]))
    methods = sorted(set(inst["method"]) | set(dly["method"]))

    ti = store.read_truth_instant()
    ti = ti[ti["temp_c"].notna()].drop_duplicates(subset=["station_id", "valid_time"])
    if len(inst) and len(ti):
        tgt = inst[["station_id", "climo_date", "_hour", "valid_time", "temp_c"]].drop_duplicates(
            subset=["station_id", "valid_time"])
        for lead in leads:
            src = ti[["station_id", "valid_time", "temp_c"]].rename(columns={"temp_c": "fcst"})
            src = src.copy()
            src["valid_time"] = src["valid_time"] + pd.Timedelta(days=lead)
            m = tgt.merge(src, on=["station_id", "valid_time"], how="inner")
            if m.empty:
                continue
            m["lead_day"] = lead
            m["err"] = m["fcst"] - m["temp_c"]
            # the pooled t2 baseline obeys the same four-instant completeness rule (§2.3)
            full = m.groupby(["station_id", "climo_date"], observed=True)["_hour"].transform(
                "nunique") == 4
            out.append(pd.concat([m[full].assign(variable="t2"),
                                  m.assign(variable=m["_hour"].map(HOUR_VARIABLE))],
                                 ignore_index=True))
    if len(dly):
        rec = dly[["station_id", "climo_date", "variable", "temp_c"]].drop_duplicates(
            subset=["station_id", "climo_date", "variable"])
        for lead in leads:
            src = rec.rename(columns={"temp_c": "fcst"}).copy()
            src["climo_date"] = src["climo_date"] + pd.Timedelta(days=lead)
            m = rec.merge(src, on=["station_id", "climo_date", "variable"], how="inner")
            if m.empty:
                continue
            m["lead_day"] = lead
            m["_hour"] = 0
            m["err"] = m["fcst"] - m["temp_c"]
            out.append(m)
    cols = ["station_id", "model_id", "init_hour", "lead_day", "method", "climo_date", "variable",
            "_hour", "valid_time", "init_time", "err", "temp_c"]
    if not out:
        return pd.DataFrame(columns=cols)
    base = pd.concat(out, ignore_index=True)
    base["model_id"] = PERSISTENCE_ID
    base["init_time"] = pd.NaT
    if "valid_time" not in base:
        base["valid_time"] = pd.NaT
    parts = []
    for ih in init_hours:
        for meth in methods:
            p = base.copy()
            p["init_hour"] = ih
            p["method"] = meth
            parts.append(p)
    return pd.concat(parts, ignore_index=True)[cols]


def window_slice(rows: pd.DataFrame, window: str, as_of: pd.Timestamp) -> pd.DataFrame:
    if window == "all":
        return rows
    days = int(window.rstrip("d"))
    return rows[rows["climo_date"] >= as_of - pd.Timedelta(days=days - 1)]


def _unit_days(sub: pd.DataFrame, is_all: bool) -> pd.DataFrame:
    """Collapse to one row per scored day: mean over instants, then (for ALL) over stations."""
    per = (
        sub.groupby(["station_id", "climo_date"], observed=True)["err"]
        .agg(a=lambda s: s.abs().mean(), s="mean",
             q=lambda s: (s ** 2).mean(),
             h1=lambda s: (s.abs() <= HIT_THRESHOLDS_C[0] + 1e-9).mean())
        .reset_index()
    )
    if is_all:
        per = per.groupby("climo_date", observed=True)[["a", "s", "q", "h1"]].mean().reset_index()
    return per


def part1() -> int:
    scores, _ = store.read_scores()
    if scores.empty:
        print("no scores in data/ — run `castcheck derive && castcheck verify` first")
        return 1
    inst = naive_instant_rows()
    dly = naive_daily_rows()
    pers = naive_persistence_rows(inst, dly)
    rows = pd.concat([r for r in (inst, dly, pers) if len(r)], ignore_index=True)
    as_of = rows["climo_date"].max()

    cand = scores[scores["n"] >= 5]
    picks = pd.concat([
        cand[(cand["station_id"] != "ALL") & (cand["variable"] == "t2")
             & (cand["model_id"] != PERSISTENCE_ID)].head(1),
        cand[(cand["station_id"] != "ALL") & (cand["variable"] == "t2_18z")
             & (cand["model_id"] != PERSISTENCE_ID)].head(1),
        cand[(cand["station_id"] != "ALL") & (cand["variable"] == "tmax_s")
             & (cand["model_id"] != PERSISTENCE_ID)].head(1),
        cand[(cand["station_id"] == "ALL") & (cand["variable"] == "t2")].head(1),
        cand[(cand["model_id"] == PERSISTENCE_ID) & (cand["variable"] == "t2")].head(1),
    ])
    if picks.empty:
        print("no comparable score rows found")
        return 1
    bad = 0
    print(f"{'group':<80} {'stat':<6} {'published':>12} {'naive':>12} {'Δ':>10}")
    print("-" * 126)
    for _, r in picks.iterrows():
        sub = rows[(rows["model_id"] == r["model_id"])
                   & (rows["init_hour"] == r["init_hour"])
                   & (rows["lead_day"] == r["lead_day"])
                   & (rows["variable"] == r["variable"])
                   & (rows["method"] == r["method"])]
        if r["station_id"] != "ALL":
            sub = sub[sub["station_id"] == r["station_id"]]
        # METHODOLOGY §7: scores stop at the latest model_version segment, which the row publishes
        seg = pd.to_datetime(r["segment_start"], utc=True, errors="coerce")
        if pd.notna(seg) and r["model_id"] != PERSISTENCE_ID:
            sub = sub[pd.to_datetime(sub["init_time"], utc=True) >= seg]
        per_day = window_slice(_unit_days(sub, r["station_id"] == "ALL"), r["window"], as_of)
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
            print(f"{label:<80} {stat:<6} {pub:12.6f} {val:12.6f} {delta:10.2e}{flag}")
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
    win = win[win.index.hour.isin(INSTANT_HOURS)]
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


#: The statistics `consistency.yml` requires to be reproducible. `computed_at`, `period_end` and the
#: bootstrap CI columns are deliberately absent: the first two are timestamps and the CIs come from a
#: seeded resample of the *same* days, so they follow from `n` and the errors — if a CI moved without
#: one of these moving, the seed derivation changed, which `test_verify.py` is the right place for.
COMPARE_STATS = ["n", "n_debiased", "n_common", "mae", "bias", "rmse", "hit1f", "hit2f", "hit3f",
                 "mae_debiased", "mae_persistence_common", "skill_persistence"]


def _close(a: float, b: float, tol: float = TOL) -> bool:
    """Equal within `tol`, scaled by magnitude; two NaNs count as equal, NaN vs a number does not."""
    na, nb = pd.isna(a), pd.isna(b)
    if na or nb:
        return bool(na and nb)
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def part3(incremental: Path, tol: float = TOL) -> int:
    """Compare the just-recomputed full ``data/scores/latest.parquet`` against a saved incremental one."""
    full, _ = store.read_scores()
    if full.empty:
        print("no scores in data/ — run `castcheck derive --full && castcheck verify` first")
        return 1
    inc = pd.read_parquet(incremental)
    print(f"incremental: {len(inc):>8} rows  ({incremental})")
    print(f"full:        {len(full):>8} rows  (data/scores/latest.parquet)\n")

    stats = [c for c in COMPARE_STATS if c in full.columns and c in inc.columns]
    a = inc.set_index(KEY)[stats].sort_index()
    b = full.set_index(KEY)[stats].sort_index()
    only_inc, only_full = a.index.difference(b.index), b.index.difference(a.index)
    both = a.index.intersection(b.index)
    bad = 0

    # A row that exists on one side only is a real inconsistency: with the same data/ in front of
    # them, the two derivation paths must produce the same table, not merely agree where they overlap.
    for label, idx in (("only in the incremental scores", only_inc), ("only in the full recompute", only_full)):
        if len(idx):
            bad += len(idx)
            print(f"{len(idx)} row(s) {label}:")
            for k in list(idx)[:10]:
                print(f"  {'/'.join(str(x) for x in k)}")
            if len(idx) > 10:
                print(f"  …and {len(idx) - 10} more")
            print()

    worst: dict[str, tuple[float, str]] = {}
    for stat in stats:
        sa, sb = a.loc[both, stat], b.loc[both, stat]
        for k, va, vb in zip(both, sa.to_numpy(), sb.to_numpy()):
            if _close(va, vb, tol):
                continue
            bad += 1
            d = float("inf") if (pd.isna(va) or pd.isna(vb)) else abs(float(va) - float(vb))
            if stat not in worst or d > worst[stat][0]:
                worst[stat] = (d, f"{'/'.join(str(x) for x in k)}  incremental={va!r} full={vb!r}")

    print(f"{'stat':<24} {'compared':>10} {'mismatches':>11}   worst")
    print("-" * 110)
    for stat in stats:
        n_bad = sum(1 for k, va, vb in zip(both, a.loc[both, stat].to_numpy(), b.loc[both, stat].to_numpy())
                    if not _close(va, vb, tol))
        note = f"Δ={worst[stat][0]:.3e}  {worst[stat][1]}" if stat in worst else ""
        print(f"{stat:<24} {len(both):>10} {n_bad:>11}   {note}")
    print(f"\n{'FAILED' if bad else 'OK'}: {bad} difference(s) beyond {tol:g} between the incremental "
          f"and the full recompute")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-network", action="store_true", help="skip the Open-Meteo magnitude check")
    ap.add_argument("--compare-incremental", metavar="OLD.parquet", default="",
                    help="incremental-vs-full mode: compare data/scores/latest.parquet with this file "
                         "instead of running the naive recomputation")
    ap.add_argument("--tol", type=float, default=TOL, help=f"absolute tolerance (default {TOL:g})")
    args = ap.parse_args()
    if args.compare_incremental:
        raise SystemExit(part3(Path(args.compare_incremental), args.tol))
    rc = part1()
    if not args.no_network:
        part2()
    raise SystemExit(rc)
