# CastCheck

**Independent, daily, station-level verification of public weather forecasts.**

Every day CastCheck takes the *raw* 2 m temperature forecasts of operational NWP models (ECMWF IFS HRES, NCEP GFS) and AI models (ECMWF AIFS Single; NOAA/CIRA operational runs of GraphCast, Pangu-Weather, FourCastNet v2 and Aurora from both GFS and IFS initial conditions), extracts them at 23 U.S. airport stations, and scores them against observations. Every *station × model × lead day* gets a permanent URL with confidence intervals; the full history is an open dataset.

The headline metric (methodology v0.3) is **`t2`**: the instantaneous 2 m temperature at 00/06/12/18 UTC, verified against the observation at the *same* instants, pooled over the four. Daily extremes are published two ways — `tmax_s`/`tmin_s`, the max/min of the four forecast samples against the max/min of the four *observed* samples (like for like), and `tmax_cli`/`tmin_cli`, the same forecast samples against the true NWS Daily Climate Report extremes. The second carries a sampling penalty whose size depends on each model's own diurnal amplitude, so it is published for operational relevance and never used for ranking.

- Site: https://castcheck.zifanzhang.com
- Methodology: [METHODOLOGY.md](METHODOLOGY.md) (versioned with the data)
- Data: Hugging Face `castcheck/temperature-verification` (primary), Kaggle mirror
- API: `https://castcheck.zifanzhang.com/api/v1/...` (static JSON)
- Citation metadata: [CITATION.cff](CITATION.cff); every built page carries the source commit in its footer

> **Fairness statement.** These are raw model outputs on the native 0.25° grid, without MOS, bias correction, downscaling or any post-processing. They are not equivalent to the products end users receive, and the scores here understate operational forecast quality. Every model is sampled at the same four instants, and the headline metric compares those samples with observations at the same instants, so no part of it depends on a model's own diurnal amplitude. Where the comparison is against the true daily extremes instead (`tmax_cli`/`tmin_cli`), it carries a sampling penalty that is *not* equal across models — its size depends on each model's diurnal amplitude — which is why those numbers are labelled secondary and are never ranked.

## How it works

```
ECMWF Open Data (.index byte-range)  ┐
AWS AIWP (remote lazy NetCDF)        ├─▶ station values ─▶ daily extremes per lead day ─┐
AWS GFS (.idx byte-range)            ┘                                                   ├─▶ scores + bootstrap CIs ─▶ site / API / datasets
IEM ASOS (routine METAR at 00/06/12/18 UTC) ─▶ truth_instant ─────────────────────────────┤
NWS CLI (first final report), CF6, hourly obs, IEM archive ─▶ truth_daily with QC flags ──┘
```

## Published tables

| table | one row per | what it is |
|---|---|---|
| `forecast_values` | model run × valid time × station × variable × method | the raw 6-hourly instantaneous layer, before any daily extreme is taken — the table to start from to re-derive or check anything |
| `truth_instant` | station × valid time (00/06/12/18 UTC) | the observed 2 m temperature at the common instants: the routine METAR nearest the hour within ±35 min (`station_id, valid_time, temp_c, obs_time, source, n_reports, qc_flag`). Truth for `t2*` and for `tmax_s`/`tmin_s` |
| `truth_daily` | station × climatological day × source | first-final NWS CLI daily extremes with QC flags and stored (never scored) corrections. Truth for `tmax_cli`/`tmin_cli` |
| `daily_forecasts` | model run × station × climatological day | sampled and native daily extremes per lead day |
| `scores` | station × model × init × lead × variable × method × window | the published aggregates with bootstrap intervals, skill and sample sizes |
| `pairwise` | the same, per model pair | paired MAE differences on common days, with `p_boot`, `distinguishable_uncorrected` and `distinguishable_holm` |

Column-by-column definitions with units are on [/data/](https://castcheck.zifanzhang.com/data/#schema).

All logic lives in importable, tested functions; `castcheck` is a thin CLI over them. Pipelines run on GitHub Actions (with a local launchd mirror) and deploy a static site to Cloudflare Pages. See [DESIGN.md](DESIGN.md).

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,publish]"

.venv/bin/castcheck fetch --model ifs_hres --init 2026-08-30T00   # one run, all stations
.venv/bin/castcheck truth --date 2026-08-29                       # observations and NWS climate reports
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

Please cite as: *CastCheck, methodology v0.3, https://castcheck.zifanzhang.com (accessed YYYY-MM-DD).*
Machine-readable metadata is in [CITATION.cff](CITATION.cff).
The methodology version that produced a given table is in its own `methodology_version` column, and in
`castcheck.METHODOLOGY_VERSION`; cite the version you actually used.
