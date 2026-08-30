# CastCheck — Engineering Design (v0.1)

Read `METHODOLOGY.md` first; this document turns it into code contracts. Implementers (human or agent) must not change the contracts below without editing this file in the same change.

## 0. Principles

- **Deterministic and replayable.** Every published number must be recomputable from the archived station values + truth + this repo at a given commit.
- **Explicit missing.** A failed fetch or an absent field produces a row with `missing_reason`, never a silently shorter table.
- **Raw archives are upstream's job.** We do not store GRIB/NetCDF. We store extracted station values (small) and everything derived from them.
- **Pure functions + thin CLI.** All logic in importable functions with docstrings and tests; `castcheck/cli.py` only wires them.
- **Time is UTC in storage.** `pandas` `datetime64[ns, UTC]`. Local/LST conversions happen only in `castcheck/climo_day.py`.
- **Python ≥ 3.11, `uv` for environments, `ruff` for lint, `pytest` for tests.** No notebooks in the repo.

## 1. Repository layout

```
castcheck/                 python package
  __init__.py
  config.py                load config/*.yaml → dataclasses (Station, ModelSpec)
  climo_day.py             LST day math: day bounds, lead_day, common-sample valid times
  grid.py                  bilinear / nearest extraction on regular lat-lon grids (numpy only)
  store.py                 parquet schemas, read/write of monthly shards, dedupe rules
  sources/
    base.py                Source protocol + FetchResult
    ecmwf.py               IFS HRES + AIFS Single via .index byte-range (data.ecmwf.int + AWS mirror)
    gfs.py                 GFS 0.25 via .idx byte-range (AWS noaa-gfs-bdp-pds / NOMADS)
    aiwp.py                AIWP NetCDF remote lazy read (h5netcdf + fsspec), all stations per layer
    nws_cli.py             CLI product discovery/parsing (api.weather.gov + IEM AFOS archive)
    nws_cf6.py             CF6 monthly parsing
    nws_obs.py             hourly station observations (fallback / QC)
  derive.py                station values → sampled/native daily extremes per (model, init, station, lead_day)
  truth.py                 assemble truth table from CLI/CF6/obs with first-final policy and qc flags
  verify.py                scores, bootstrap CIs, pairwise comparisons, persistence baseline
  site/
    build.py               static site generator (Jinja2) → public/
    templates/*.html
    assets/                css, small js (charts drawn client-side from JSON; no build step)
  api.py                   JSON export → public/api/v1/...
  status.py                data-completeness report → public/status.json + status page
  publish/
    hf.py                  push data/ shards to Hugging Face dataset repo (optional, token-gated)
    kaggle.py              Kaggle dataset mirror (optional)
    bluesky.py             daily image post (optional)
  cli.py                   typer app
config/
  stations.yaml, models.yaml
data/                      committed parquet shards (small); see §3
scripts/
  build_stations.py        fill lat/lon/elev from api.weather.gov, freeze into stations.yaml
  launchd/                 local backup schedules
tests/                     pytest; network tests marked @pytest.mark.network
.github/workflows/         cron pipelines (see §7)
public/                    generated site (gitignored)
```

## 2. Configuration contracts

```python
@dataclass(frozen=True)
class Station:
    id: str            # ICAO, e.g. "KNYC"
    name: str
    cli_pil: str       # AFOS pil, e.g. "CLINYC"
    tz: str            # IANA tz; used ONLY to derive the fixed standard offset
    std_offset_h: int  # e.g. -5; computed once in scripts/build_stations.py and frozen
    lat: float; lon: float; elev_m: float | None
    kalshi: str | None # informational

@dataclass(frozen=True)
class ModelSpec:
    model_id: str; family: str; source: str; product: str
    init_field: str | None   # AIWP only: "GFS" | "IFS"
    inits: tuple[int, ...]   # (0, 12)
    step_h: int; max_h: int
    native_extremes: tuple[str, ...]
```

`config.load_stations()` and `config.load_models()` are the only readers of the YAML files.

## 3. Data model (parquet, long format)

All tables carry `schema_version: str` (currently `"0.1"`) and `methodology_version: str`.

### 3.1 `forecast_values` — one row per extracted value
Path: `data/forecast_values/model_id=<id>/year_month=<YYYY-MM>.parquet` (partition by init month).

| column | type | notes |
|---|---|---|
| model_id | str | from models.yaml |
| model_version | str | e.g. `"GRAP_v100"`, `"ifs-cy50r1"`, `"gfs-v16"`; `"unknown"` allowed but logged |
| init_time | timestamp[UTC] | |
| valid_time | timestamp[UTC] | |
| lead_h | int16 | valid − init in hours |
| station_id | str | |
| variable | str | `t2` (instantaneous 2 m K→°C), `mx2t3`, `mn2t3`, `mx2t6`, `mn2t6`, `tmax6`, `tmin6` (GFS buckets) |
| bucket_h | int8 | window length for extreme fields (0 for instantaneous) |
| method | str | `bilinear` \| `nearest` |
| value_c | float32 | °C; NaN if missing |
| missing_reason | str | `""` if present; else e.g. `"no_file"`, `"no_field"`, `"fill_value"`, `"http_404"` |
| source_url | str | exact object/URL used |
| fetched_at | timestamp[UTC] | |

Uniqueness key: `(model_id, init_time, valid_time, station_id, variable, bucket_h, method)`. `store.upsert()` replaces on key; a later fetch with a value replaces an earlier missing row, never the reverse.

### 3.2 `truth_daily` — one row per station-day-source
Path: `data/truth_daily/year=<YYYY>.parquet`.

| column | type | notes |
|---|---|---|
| station_id | str | |
| climo_date | date | LST day |
| source | str | `CLI` \| `CF6` \| `OBS` |
| tmax_f / tmin_f | int16 | as reported (NULL if missing) |
| tmax_c / tmin_c | float32 | converted |
| issuance_time | timestamp[UTC] | for CLI: the product issuance |
| is_final | bool | CLI: YESTERDAY block of first post-midnight issuance |
| revised | bool | a later corrected CLI exists |
| revised_tmax_f / revised_tmin_f | int16 | latest correction (not used in scores) |
| qc_flag | str | `""`, `"obs_diff_gt2f"`, `"cli_missing_value"` … |
| product_id | str | api.weather.gov product id or IEM key |

### 3.3 `daily_forecasts` — derived, one row per (model, init, station, lead_day)
Path: `data/daily_forecasts/model_id=<id>/year=<YYYY>.parquet`.

| column | type |
|---|---|
| model_id, model_version, init_time, station_id | as above |
| climo_date | date |
| lead_day | int8 |
| tmax_sampled_c, tmin_sampled_c | float32 (NaN if any of 4 samples missing) |
| n_samples | int8 (0–4) |
| tmax_native_c, tmin_native_c | float32 (NaN if model has no native field) |
| method | `bilinear` \| `nearest` |
| missing_reason | str |

### 3.4 `scores` — published aggregates
Path: `data/scores/latest.parquet` (overwritten daily) + `data/scores/history/<YYYY-MM-DD>.parquet`.

`station_id, model_id, init_hour, lead_day, variable(tmax|tmin), method, window(30d|90d|365d|all), n, mae, bias, rmse, hit1f, hit2f, hit3f, skill_persistence, mae_ci_low, mae_ci_high, bias_ci_low, bias_ci_high, period_start, period_end, computed_at, methodology_version, schema_version`.

Plus `station_id='ALL'` rows aggregating over all stations (mean over stations of daily errors, same bootstrap).

### 3.5 `pairwise` — model-vs-model
`station_id, init_hour, lead_day, variable, window, model_a, model_b, n_common, mae_diff, ci_low, ci_high, significant(bool)`.

## 4. Module contracts

```python
# climo_day.py
def day_bounds_utc(station: Station, climo_date: date) -> tuple[datetime, datetime]
def common_sample_times(station: Station, climo_date: date) -> list[datetime]   # exactly 4, UTC
def lead_day(init_time: datetime, climo_date: date) -> int                       # climo_date − init UTC date
def climo_dates_for_run(station: Station, init_time: datetime, max_h: int) -> list[date]  # days fully covered

# grid.py  (regular lat-lon; lat may be descending; lon may be 0..360 or -180..180 — normalise inside)
def bilinear(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float) -> float
def nearest(field, lats, lons, lat, lon) -> float
def extract_all(field, lats, lons, stations: list[Station]) -> dict[str, tuple[float, float]]  # id → (bilinear, nearest)

# sources/base.py
@dataclass
class FetchRequest: model: ModelSpec; init_time: datetime; stations: list[Station]
@dataclass
class FetchResult: rows: pd.DataFrame  # forecast_values schema; includes explicit missing rows
class Source(Protocol):
    def available_inits(self, model: ModelSpec, start: date, end: date) -> list[datetime]
    def fetch_run(self, req: FetchRequest) -> FetchResult    # must never raise on partial failure

# derive.py
def daily_from_values(values: pd.DataFrame, stations, models) -> pd.DataFrame   # daily_forecasts
# truth.py
def assemble_truth(cli_rows, cf6_rows, obs_rows) -> pd.DataFrame                # truth_daily with first-final + qc
# verify.py
def score(daily: pd.DataFrame, truth: pd.DataFrame, windows=(30,90,365,None), n_boot=1000, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]  # scores, pairwise
def persistence_daily(truth: pd.DataFrame) -> pd.DataFrame                      # baseline in daily_forecasts schema
```

Source-specific notes (from measured behaviour, 2026-08-30):
- **ECMWF**: base URL `https://data.ecmwf.int/forecasts/{YYYYMMDD}/{HH}z/{ifs|aifs-single}/0p25/{oper}/{YYYYMMDD}{HH}0000-{step}h-{oper}-fc.grib2` with `.index` (one JSON per line, `_offset/_length`); AWS mirror `https://ecmwf-forecasts.s3.amazonaws.com/{YYYYMMDD}/{HH}z/...` (same layout, 2023-01-18→). Range GET returns 206. Longitudes are −180..179.75. `cfgrib` with `indexpath=""`. Steps: IFS 0-144 @3h, 150-360 @6h; AIFS @6h. 06/18Z IFS only to 144h (not used).
- **GFS**: `https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{YYYYMMDD}/{HH}/atmos/gfs.t{HH}z.pgrb2.0p25.f{FFF}` + `.idx`; fields `TMP:2 m above ground`, `TMAX:2 m above ground:{a}-{b} hour max fcst`, `TMIN:...`. Longitudes 0..359.75.
- **AIWP**: `https://noaa-oar-mlwp-data.s3.amazonaws.com/{PROD}_{ver}_{INIT}/{YYYY}/{MMDD}/{PROD}_{ver}_{INIT}_{YYYYMMDDHH}_f000_f240_06.nc` (e.g. `GRAP_v100_IFS`); discover `{ver}` by listing the bucket prefix (`?list-type=2&prefix=`). Open with `h5netcdf` over `fsspec` HTTP (block_size 4 MiB); variable `t2` shape (41, 721, 1440), chunk = one full layer → read one time index, extract all stations; **f000 is a fill value (9.97e36) → missing**. Longitudes 0..359.75.
- **NWS CLI**: `https://api.weather.gov/products/types/CLI/locations/{LOC}` (LOC = pil minus `CLI`), then `/products/{id}`; final report = first issuance after local midnight whose text contains the `YESTERDAY` block; parse fixed-width `MAXIMUM  76  355 PM`. Must send a real `User-Agent`. Retention ~7 days; history: `https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil={PIL}&sdate=...&edate=...&fmt=text`.
- **Obs**: `https://api.weather.gov/stations/{ICAO}/observations?start=..&end=..`; `maxTemperatureLast24Hours` is null — derive from hourly `temperature` only as fallback.

## 5. CLI (typer)

```
castcheck fetch   --model ifs_hres --init 2026-08-30T00   [--stations KNYC,KORD]
castcheck fetch-latest                       # all models, all inits that should exist by now
castcheck backfill --model gfs --start 2024-01-01 --end 2024-03-31
castcheck truth    --date 2026-08-29         # CLI first-final for all stations (+ obs fallback)
castcheck truth-backfill --start ... --end ...   # IEM
castcheck derive                              # values → daily_forecasts
castcheck verify                              # → scores, pairwise
castcheck build-site                          # → public/
castcheck status                              # → public/status.json (+ nonzero exit if gaps today)
castcheck publish hf|kaggle|bluesky          # optional, token-gated
```

All commands are idempotent and safe to re-run.

## 6. Site and API

Routes (static, under `public/`):
- `/` — overview: leaderboard for lead day 1/3/5/7 (ALL stations, 90-day window, bilinear, 00Z), availability bars, last update.
- `/station/{ICAO}/` — station page: all models × leads, 4 windows, time series of daily errors, truth vs forecasts for last 30 days.
- `/model/{model_id}/` — model page.
- `/station/{ICAO}/model/{model_id}/lead/{d}/` — the permanent link: one score card with CI, n, period, method toggles (bilinear/nearest, 00Z/12Z), the daily error series, and a citation block.
- `/methodology/` — rendered METHODOLOGY.md. `/status/` — completeness. `/data/` — download links + schema.
- `/api/v1/scores/latest.json`, `/api/v1/scores/{station}/{model}/{lead}.json`, `/api/v1/pairwise/latest.json`, `/api/v1/stations.json`, `/api/v1/models.json`, `/api/v1/status.json` — plain static JSON.

Charts: minimal inline SVG or a tiny client-side script reading the JSON; no framework, no build step. Pages must be readable with JS disabled (tables first, charts as enhancement). Theme: light, system font, one accent colour.

## 7. Pipelines (GitHub Actions; local launchd mirrors the same commands)

| workflow | schedule (UTC) | steps |
|---|---|---|
| `fetch-00z.yml` | 05:00, 06:00, 08:00, 09:30 | `fetch-latest` (each run only fetches what is new) |
| `fetch-12z.yml` | 21:00 | same |
| `truth-daily.yml` | 10:30, 16:00 | `truth --date yesterday`, second run picks up corrections |
| `verify-publish.yml` | 11:00 | `derive` → `verify` → `build-site` → `status` → `wrangler pages deploy public --project-name castcheck` → commit data shards |
| `backfill.yml` | manual dispatch | ranges per model |

Secrets: `CLOUDFLARE_API_TOKEN` (Pages edit), optional `HF_TOKEN`, `KAGGLE_*`, `BSKY_*`. Data shards are committed by the workflow with `[data]` commit messages.

## 8. Testing

- Unit tests with synthetic grids for `grid.py`, fixed-date cases for `climo_day.py` (EST/CST/MST/PST/Phoenix, DST and non-DST dates), CLI parsing fixtures (final vs intermediate report, corrected report, missing `M`), `verify.py` on a toy dataset with known MAE/bias and a deterministic bootstrap seed.
- Network tests (`@pytest.mark.network`) that fetch one real step per source for one station and assert value sanity (−60 °C < T < 60 °C) and presence of explicit missing rows on a 404.

## 9. Work split for parallel implementation

| Package | Owner | Depends on |
|---|---|---|
| A. `config.py`, `climo_day.py`, `grid.py`, `store.py`, `sources/base.py`, `scripts/build_stations.py` | core | — |
| B. `sources/ecmwf.py`, `sources/gfs.py` | NWP adapters | A (contracts only) |
| C. `sources/aiwp.py` | AI adapters | A |
| D. `sources/nws_cli.py`, `nws_cf6.py`, `nws_obs.py`, `truth.py` | truth | A |
| E. `derive.py`, `verify.py`, `api.py`, `status.py`, `site/` | derive/verify/site | A + schemas |
| F. `cli.py`, workflows, launchd, `publish/` | ops | all |

Owners of B–E may stub `Station`/`ModelSpec` locally if A is not merged yet, but must match §2 exactly.
