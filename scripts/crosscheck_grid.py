"""Independent check of the station extraction (METHODOLOGY §2.6).

Two questions:

1. Does :func:`castcheck.grid.bilinear` give the *same* answer for the same physical field stored in
   the ECMWF convention (latitude descending, longitude −180..179.75) and in the GFS/AIWP convention
   (latitude descending, longitude 0..359.75)?  Checked exactly, on a synthetic planar field and on a
   spherical-harmonic-like field, at every configured station.
2. On the real archive: how far apart are the f006 values of models that share an initial field?
   A longitude sign or wrap bug would show up as a handful of stations several kelvin out, not as a
   smooth O(0.5 K) spread.

Run with ``PYTHONPATH=. .venv/bin/python scripts/crosscheck_grid.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from castcheck import store
from castcheck.config import load_stations
from castcheck.grid import bilinear, nearest

PAIRS = [("graphcast_ifs", "ifs_hres"), ("aurora_ifs", "ifs_hres"),
         ("fourcastnet_ifs", "ifs_hres"), ("pangu_gfs", "gfs")]


def _grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = np.linspace(90.0, -90.0, 721)          # descending, both conventions
    lon_ecmwf = np.arange(-180.0, 180.0, 0.25)    # −180..179.75
    lon_gfs = np.arange(0.0, 360.0, 0.25)         # 0..359.75
    return lats, lon_ecmwf, lon_gfs


def _field(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """A smooth field that is a genuine function of position on the sphere, so the two longitude
    conventions must sample exactly the same numbers."""
    la = np.deg2rad(lats)[:, None]
    lo = np.deg2rad(lons)[None, :]
    return (288.0 + 20.0 * np.sin(la) + 3.0 * np.cos(3 * lo) * np.cos(la)
            + 1.5 * np.sin(2 * lo + 0.7) * np.sin(2 * la))


def synthetic_check() -> None:
    lats, lon_e, lon_g = _grids()
    fe = _field(lats, lon_e)
    fg = _field(lats, lon_g)
    worst_bl = worst_nn = 0.0
    for s in load_stations():
        b_e = bilinear(fe, lats, lon_e, s.lat, s.lon)
        b_g = bilinear(fg, lats, lon_g, s.lat, s.lon)
        n_e = nearest(fe, lats, lon_e, s.lat, s.lon)
        n_g = nearest(fg, lats, lon_g, s.lat, s.lon)
        worst_bl = max(worst_bl, abs(b_e - b_g))
        worst_nn = max(worst_nn, abs(n_e - n_g))
    print(f"synthetic 0.25° field, {len(load_stations())} stations:")
    print(f"  max |bilinear(ECMWF grid) − bilinear(GFS grid)| = {worst_bl:.3e} K")
    print(f"  max |nearest (ECMWF grid) − nearest (GFS grid)|  = {worst_nn:.3e} K")

    # wrap-around: a point between the last and first longitude node of each convention
    for name, lons, field in (("−180..179.75", lon_e, fe), ("0..359.75", lon_g, fg)):
        for lon in (-0.125, 179.875, -179.875, 359.875 - 360.0):
            v = bilinear(field, lats, lons, 40.0, lon)
            exact = float(_field(np.array([40.0]), np.array([lon % 360.0]))[0, 0])
            print(f"  wrap {name:<12} lon={lon:>9.3f}: bilinear={v:8.4f}  exact={exact:8.4f}"
                  f"  diff={v - exact:+.4f}")


def real_data_check() -> None:
    fv = store.read_forecast_values()
    if fv.empty:
        print("\nno forecast_values in data/ — skipping the real-data check")
        return
    t2 = fv[(fv["variable"] == "t2") & (fv["missing_reason"] == "") & (fv["method"] == "bilinear")]
    t2 = t2.assign(lead_h=pd.to_numeric(t2["lead_h"]))
    f6 = t2[t2["lead_h"] == 6]
    inits = sorted(set.intersection(*[
        set(f6.loc[f6["model_id"] == m, "init_time"].unique()) for pair in PAIRS for m in pair
        if (f6["model_id"] == m).any()
    ] or [set()]))
    if not inits:
        print("\nno common initialization with f006 for every model — skipping")
        return
    init = inits[-1]
    piv = f6[f6["init_time"] == init].pivot_table(
        index="station_id", columns="model_id", values="value_c"
    )
    print(f"\nreal archive, init {pd.Timestamp(init)}, f006 bilinear, {len(piv)} stations")
    print(f"{'pair':<32} {'n':>3} {'mean':>7} {'sd':>6} {'min':>7} {'max':>7} {'med|d|':>7}")
    for a, b in PAIRS:
        if a not in piv or b not in piv:
            continue
        d = (piv[a] - piv[b]).dropna()
        print(f"{a + ' − ' + b:<32} {len(d):>3} {d.mean():7.3f} {d.std():6.3f} "
              f"{d.min():7.3f} {d.max():7.3f} {d.abs().median():7.3f}")
    print("\nAIWP f006 is a 6 h AI forecast from the same analysis, not the analysis itself, so a"
          "\nsub-kelvin spread is physics, not a grid bug; a sign/wrap error would put single"
          "\nstations 5–20 K out.")


if __name__ == "__main__":
    synthetic_check()
    real_data_check()
