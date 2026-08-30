"""Fill lat/lon/elev_m/std_offset_h in config/stations.yaml from api.weather.gov, then freeze.

Run once (or when adding stations):  .venv/bin/python scripts/build_stations.py
Existing coordinates are never overwritten (pass --force to refresh).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from castcheck.config import CONFIG_DIR, USER_AGENT, standard_offset_hours

FORCE = "--force" in sys.argv
path = CONFIG_DIR / "stations.yaml"
doc = yaml.safe_load(path.read_text(encoding="utf-8"))
sess = requests.Session()
sess.headers["User-Agent"] = USER_AGENT

for s in doc["stations"]:
    s["std_offset_h"] = standard_offset_hours(s["tz"])
    if not FORCE and all(k in s for k in ("lat", "lon", "elev_m")):
        continue
    r = sess.get(f"https://api.weather.gov/stations/{s['id']}", timeout=30)
    if r.status_code != 200:
        print(f"{s['id']}: HTTP {r.status_code} — left empty", file=sys.stderr)
        continue
    j = r.json()
    lon, lat = j["geometry"]["coordinates"]
    elev = j["properties"].get("elevation", {}).get("value")
    s["lat"], s["lon"] = round(float(lat), 5), round(float(lon), 5)
    s["elev_m"] = round(float(elev), 1) if elev is not None else None
    print(f"{s['id']}: lat={s['lat']} lon={s['lon']} elev={s['elev_m']} tz={s['tz']} std={s['std_offset_h']}")
    time.sleep(0.3)

header = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#"))
body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=200)
path.write_text(header + "\n" + body, encoding="utf-8")
print("frozen ->", path)
