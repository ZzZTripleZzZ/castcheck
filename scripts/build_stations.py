"""Fill the frozen station metadata in config/stations.yaml, then rewrite the file.

Three passes, each opt-in beyond the first:

* default          lat / lon / elev_m / std_offset_h from api.weather.gov (never overwritten
                   without --force, because the whole point of freezing coordinates is that a
                   later upstream edit cannot silently move a station).
* ``--iem-id``     verify the station's id in the IEM ASOS archive by asking for one day of data.
* ``--grid-elev``  mean surface elevation of the 0.25 deg model cell containing the station, from
                   ETOPO 2022 (see below), plus the derived ``dz_m``.

Run:  .venv/bin/python scripts/build_stations.py [--force] [--iem-id] [--grid-elev]

Representative elevation (DESIGN §10.4, external review B7)
-----------------------------------------------------------
A model cannot forecast a station's temperature better than its own orography allows: KDEN sits at
1647 m and the 0.25 deg cell around it averages ~1680 m, and KSFO's cell is largely ocean. Publishing
``dz_m`` next to each score turns "some stations are just harder" from a caveat into a number, and
``|dz_m| x 6.5 K/km`` is its first-order size.

Source: **ETOPO 2022 60 arc-second surface-elevation grid**, NOAA National Centers for Environmental
Information (doi:10.25921/fd45-gt74). It is a US Government work in the public domain, so
redistributing derived values carries no licence condition. Cells are read over OPeNDAP from NCEI's
THREDDS server, a few hundred values per station, so nothing has to be downloaded in bulk.

The 0.25 deg cell is the one the forecast grids use: centres on exact multiples of 0.25 deg, so the
cell spans centre +/- 0.125 deg. ETOPO cells are averaged over that box weighted by the fraction of
each cell inside it and by cos(latitude), which is the cell's area.
"""

from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from castcheck.config import CONFIG_DIR, USER_AGENT, standard_offset_hours

ETOPO = ("https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
         "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc")
#: ETOPO 2022 60" is a 1 arc-minute grid; cell (i, j) is centred at
#: (-90 + (i + 0.5)/60, -180 + (j + 0.5)/60) degrees.
ETOPO_PER_DEG = 60
ETOPO_NLAT, ETOPO_NLON = 10800, 21600

GRID_DEG = 0.25  # the model grid these values describe

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

FORCE = "--force" in sys.argv
WANT_IEM = "--iem-id" in sys.argv
WANT_GRID = "--grid-elev" in sys.argv

sess = requests.Session()
sess.headers["User-Agent"] = USER_AGENT


# --------------------------------------------------------------------------- ETOPO over OPeNDAP


def _dap_ascii(query: str) -> str:
    r = sess.get(f"{ETOPO}.ascii", params=query, timeout=120)
    r.raise_for_status()
    return r.text


def _parse_dap_grid(text: str, nrows: int) -> list[list[float]]:
    """Pull the numeric block out of an OPeNDAP ``.ascii`` Grid response.

    The payload is ``[row], v, v, ...`` lines after a dashed separator, followed by the map
    vectors; only the array rows are wanted, and there are exactly `nrows` of them.
    """
    rows: list[list[float]] = []
    for line in text.splitlines():
        m = re.match(r"^\[(\d+)\],\s*(.+)$", line.strip())
        if not m:
            continue
        rows.append([float(v) for v in m.group(2).split(",")])
        if len(rows) == nrows:
            break
    if len(rows) != nrows:
        raise RuntimeError(f"expected {nrows} rows from OPeNDAP, parsed {len(rows)}")
    return rows


def _cell_overlap(lo: float, hi: float, per_deg: int, offset: float, n: int) -> tuple[int, list[float]]:
    """First index and per-cell overlap widths (in cells) for a grid covering ``[lo, hi]``.

    ``offset`` is the coordinate of the grid's origin edge (-90 for latitude, -180 for longitude).
    Edge cells straddle the box boundary and are weighted by the fraction that falls inside it, so
    the mean is a genuine area average rather than one that silently shifts the box by half a cell.
    """
    first = max(0, math.floor((lo - offset) * per_deg))
    last = min(n - 1, math.ceil((hi - offset) * per_deg) - 1)
    widths = []
    for k in range(first, last + 1):
        c_lo = offset + k / per_deg
        c_hi = offset + (k + 1) / per_deg
        widths.append(max(0.0, min(hi, c_hi) - max(lo, c_lo)) * per_deg)
    return first, widths


def grid_cell_elevation(lat: float, lon: float) -> tuple[float, float, float]:
    """Mean ETOPO surface elevation of the 0.25 deg cell holding ``(lat, lon)``.

    Returns ``(grid_elev_m, cell_centre_lat, cell_centre_lon)``.
    """
    c_lat = round(lat / GRID_DEG) * GRID_DEG
    c_lon = round(lon / GRID_DEG) * GRID_DEG
    lat_lo, lat_hi = c_lat - GRID_DEG / 2, c_lat + GRID_DEG / 2
    lon_lo, lon_hi = c_lon - GRID_DEG / 2, c_lon + GRID_DEG / 2

    i0, w_lat = _cell_overlap(lat_lo, lat_hi, ETOPO_PER_DEG, -90.0, ETOPO_NLAT)
    j0, w_lon = _cell_overlap(lon_lo, lon_hi, ETOPO_PER_DEG, -180.0, ETOPO_NLON)
    text = _dap_ascii(f"z[{i0}:1:{i0 + len(w_lat) - 1}][{j0}:1:{j0 + len(w_lon) - 1}]")
    z = _parse_dap_grid(text, len(w_lat))

    total = 0.0
    weight = 0.0
    for i, wi in enumerate(w_lat):
        centre = -90.0 + (i0 + i + 0.5) / ETOPO_PER_DEG
        band = wi * math.cos(math.radians(centre))  # a cell's area shrinks towards the pole
        for j, wj in enumerate(w_lon):
            w = band * wj
            total += w * z[i][j]
            weight += w
    return total / weight, c_lat, c_lon


# --------------------------------------------------------------------------- IEM station id


def iem_station_id(icao: str) -> str | None:
    """The archive id that actually returns data for `icao`: the de-K'd form, else the ICAO."""
    for candidate in ([icao[1:], icao] if icao.startswith("K") and len(icao) == 4 else [icao]):
        url = (f"{IEM_ASOS}?station={candidate}&data=tmpf&year1=2025&month1=6&day1=1"
               "&year2=2025&month2=6&day2=2&tz=Etc/UTC&format=onlycomma&latlon=no"
               "&missing=M&trace=T&report_type=3")
        try:
            r = sess.get(url, timeout=60)
        except requests.RequestException as exc:
            print(f"{icao}: IEM request failed ({type(exc).__name__})", file=sys.stderr)
            return None
        if r.status_code == 200 and len(r.text.strip().splitlines()) > 1:
            return candidate
        time.sleep(0.7)
    return None


# --------------------------------------------------------------------------- main


def main() -> int:
    path = CONFIG_DIR / "stations.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    for s in doc["stations"]:
        s["std_offset_h"] = standard_offset_hours(s["tz"])
        if "kalshi" in s:  # pre-v0.3 spelling
            s["market_city"] = s.pop("kalshi")

        if FORCE or not all(k in s for k in ("lat", "lon", "elev_m")):
            r = sess.get(f"https://api.weather.gov/stations/{s['id']}", timeout=30)
            if r.status_code != 200:
                print(f"{s['id']}: HTTP {r.status_code} — left empty", file=sys.stderr)
            else:
                j = r.json()
                lon, lat = j["geometry"]["coordinates"]
                elev = j["properties"].get("elevation", {}).get("value")
                s["lat"], s["lon"] = round(float(lat), 5), round(float(lon), 5)
                s["elev_m"] = round(float(elev), 1) if elev is not None else None
                print(f"{s['id']}: lat={s['lat']} lon={s['lon']} elev={s['elev_m']} "
                      f"tz={s['tz']} std={s['std_offset_h']}")
            time.sleep(0.3)

        if WANT_IEM and (FORCE or not s.get("iem_id")):
            found = iem_station_id(s["id"])
            if found is None:
                print(f"{s['id']}: no IEM ASOS id found", file=sys.stderr)
            else:
                s["iem_id"] = found
                print(f"{s['id']}: iem_id={found}")
            time.sleep(0.7)

        if WANT_GRID and (FORCE or s.get("grid_elev_m") is None):
            if s.get("lat") is None or s.get("lon") is None:
                print(f"{s['id']}: no coordinates — grid elevation skipped", file=sys.stderr)
                continue
            try:
                gz, c_lat, c_lon = grid_cell_elevation(float(s["lat"]), float(s["lon"]))
            except Exception as exc:  # noqa: BLE001 — one station must not abort the pass
                print(f"{s['id']}: ETOPO failed ({type(exc).__name__}: {exc})", file=sys.stderr)
                continue
            s["grid_elev_m"] = round(gz, 1)
            dz = round(float(s["elev_m"]) - gz, 1) if s.get("elev_m") is not None else None
            # written out as well as derived in code: stations.csv and the site read the file, and
            # DESIGN §10.4 describes the column as part of the published station metadata
            s["dz_m"] = dz
            lapse = "" if dz is None else f" dz_m={dz} lapse={abs(dz) * 6.5 / 1000:.2f}K"
            print(f"{s['id']}: cell ({c_lat:+.2f},{c_lon:+.2f}) grid_elev_m={s['grid_elev_m']}{lapse}")
            time.sleep(0.3)

    header = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                       if line.startswith("#"))
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=200)
    path.write_text(header + "\n" + body, encoding="utf-8")
    print("frozen ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
