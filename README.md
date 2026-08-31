# CastCheck

**Independent, daily, station-level verification of public weather forecasts.**

Every day CastCheck takes the *raw* 2 m temperature forecasts of operational NWP models (ECMWF IFS HRES, NCEP GFS) and AI models (ECMWF AIFS Single; NOAA/CIRA operational runs of GraphCast, Pangu-Weather, FourCastNet v2 and Aurora from both GFS and IFS initial conditions), extracts them at 23 U.S. airport stations, and scores them against the NWS Daily Climate Report. Every *station × model × lead day* gets a permanent URL with confidence intervals; the full history is an open dataset.

- Site: https://castcheck.zifanzhang.com
- Methodology: [METHODOLOGY.md](METHODOLOGY.md) (versioned with the data)
- Data: Hugging Face `castcheck/temperature-verification` (primary), Kaggle mirror
- API: `https://castcheck.zifanzhang.com/api/v1/...` (static JSON)

> **Fairness statement.** These are raw model outputs on the native 0.25° grid, without MOS, bias correction, downscaling or any post-processing. They are not equivalent to the products end users receive, and the scores here understate operational forecast quality. Daily extremes are computed identically for every model from the common 6-hourly instantaneous samples, so the well-known under-sampling of the diurnal cycle affects all models equally.

## How it works

```
ECMWF Open Data (.index byte-range)  ┐
AWS AIWP (remote lazy NetCDF)        ├─▶ station values ─▶ daily extremes per lead day ─┐
AWS GFS (.idx byte-range)            ┘                                                   ├─▶ scores + bootstrap CIs ─▶ site / API / datasets
NWS CLI (first final report), CF6, hourly obs, IEM archive ─▶ truth with QC flags ───────┘
```

All logic lives in importable, tested functions; `castcheck` is a thin CLI over them. Pipelines run on GitHub Actions (with a local launchd mirror) and deploy a static site to Cloudflare Pages. See [DESIGN.md](DESIGN.md).

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,publish]"

.venv/bin/castcheck fetch --model ifs_hres --init 2026-08-30T00   # one run, all stations
.venv/bin/castcheck truth --date 2026-08-29                       # NWS climate reports
.venv/bin/castcheck daily                                         # derive → verify → build public/
.venv/bin/castcheck status                                        # completeness; exit 1 if today has gaps

.venv/bin/python -m pytest -q -m "not network"                    # tests (drop -m to hit the network)
uvx ruff@0.16.5 check --no-cache .                                 # lint, exactly what CI runs
```

Every command logs to stderr (`-v` for DEBUG, `CASTCHECK_LOG_JSON=1` for JSON lines), ends with a
one-line summary, and records its outcome in `data/raw/last_run.json`. The optional publishers are
token-gated and all support `--dry-run`:

```bash
.venv/bin/castcheck publish bluesky --dry-run   # writes data/raw/bluesky_preview.png + the post text
.venv/bin/castcheck publish hf --dry-run        # lists what would be uploaded, contacts nothing
```

## Data licences

ECMWF Open Data (CC-BY-4.0) · NOAA/NCEP GFS and NWS products (public domain) · NOAA/CIRA AIWP (open data) · Iowa Environmental Mesonet AFOS archive. CastCheck's published tables are CC-BY-4.0; code is MIT.

Please cite as: *CastCheck, methodology v0.2, https://castcheck.zifanzhang.com (accessed YYYY-MM-DD).*
The methodology version that produced a given table is in its own `methodology_version` column, and in
`castcheck.METHODOLOGY_VERSION`; cite the version you actually used.
