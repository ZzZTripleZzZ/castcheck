"""Station extraction on regular latitude–longitude grids (METHODOLOGY §2.6).

Works for latitude arrays in either order and longitude arrays in 0..360 or -180..180 convention.
Pure numpy; no xarray dependency so it can be unit-tested with synthetic fields.
"""

from __future__ import annotations

import numpy as np

from .config import Station


def _prep(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lon360 = bool(lons.max() > 180.0)
    return lats, lons, lon360


def _norm_lon(lon: float, lon360: bool) -> float:
    return lon % 360.0 if lon360 else ((lon + 180.0) % 360.0) - 180.0


def _bracket(coords: np.ndarray, x: float, periodic_span: float | None = None) -> tuple[int, int, float]:
    """Return indices (i0, i1) bracketing x and the fractional weight of i1. Handles descending arrays
    and periodic wrap in longitude."""
    asc = coords[1] > coords[0]
    c = coords if asc else coords[::-1]
    n = len(c)
    if periodic_span is not None and (x < c[0] or x > c[-1]):
        # wrap between last and first node
        i_lo, i_hi = n - 1, 0
        span = (c[0] + periodic_span) - c[-1]
        w = ((x - c[-1]) % periodic_span) / span
    else:
        x = min(max(x, c[0]), c[-1])
        i_hi = int(np.searchsorted(c, x, side="right"))
        i_hi = min(max(i_hi, 1), n - 1)
        i_lo = i_hi - 1
        w = (x - c[i_lo]) / (c[i_hi] - c[i_lo]) if c[i_hi] != c[i_lo] else 0.0
    if not asc:
        i_lo, i_hi = n - 1 - i_lo, n - 1 - i_hi
    return i_lo, i_hi, float(w)


def bilinear(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float) -> float:
    """Bilinear interpolation of a 2-D field[lat, lon] to (lat, lon)."""
    lats, lons, lon360 = _prep(lats, lons)
    x = _norm_lon(lon, lon360)
    j0, j1, wy = _bracket(lats, lat)
    i0, i1, wx = _bracket(lons, x, periodic_span=360.0)
    f = np.asarray(field, dtype=float)
    v = (
        f[j0, i0] * (1 - wx) * (1 - wy)
        + f[j0, i1] * wx * (1 - wy)
        + f[j1, i0] * (1 - wx) * wy
        + f[j1, i1] * wx * wy
    )
    return float(v)


def nearest(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float) -> float:
    lats, lons, lon360 = _prep(lats, lons)
    x = _norm_lon(lon, lon360)
    j = int(np.argmin(np.abs(lats - lat)))
    dlon = np.abs(((lons - x) + 180.0) % 360.0 - 180.0)
    i = int(np.argmin(dlon))
    return float(np.asarray(field, dtype=float)[j, i])


def extract_all(
    field: np.ndarray, lats: np.ndarray, lons: np.ndarray, stations: list[Station]
) -> dict[str, tuple[float, float]]:
    """id -> (bilinear, nearest). Stations without coordinates are skipped."""
    # Convert the (possibly float32, 721x1440) field exactly once; bilinear/nearest then see float64 and
    # their internal np.asarray(dtype=float) is a no-op. Measured: 121 s -> ~0 s CPU per 41-layer run.
    f = np.asarray(field, dtype=float)
    la = np.asarray(lats, dtype=float)
    lo = np.asarray(lons, dtype=float)
    out: dict[str, tuple[float, float]] = {}
    for s in stations:
        if s.lat is None or s.lon is None:
            continue
        out[s.id] = (bilinear(f, la, lo, s.lat, s.lon), nearest(f, la, lo, s.lat, s.lon))
    return out


def is_fill(value: float) -> bool:
    """AIWP uses 9.97e36 at f000; GRIB missing is typically 9999 or nan."""
    return not np.isfinite(value) or abs(value) > 1e30 or value > 400.0 or value < 100.0  # Kelvin sanity
