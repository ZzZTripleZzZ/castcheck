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
  commit_data.sh           commit + push data/ from a workflow, retrying the rebase/push race
  run_daily.sh             local mirror of the daily pipeline (launchd)
  backfill_local.sh, truth_backfill_local.sh   detached local backfills
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
Path: `data/scores/latest.parquet` (overwritten daily), `data/scores/pairwise_latest.parquet`, and a
dated snapshot `data/scores/history/<YYYY-MM-DD>.parquet`.

The authoritative column list is `castcheck.verify.SCORE_COLUMNS`; this table must be edited in the
same change as that constant.

`station_id, model_id, init_hour, lead_day, variable(tmax|tmin), method, window(30d|90d|365d|all),
n, n_stations, n_flagged, mae, bias, rmse, hit1f, hit2f, hit3f, mae_debiased, skill_persistence,
skill_persistence_debiased, mae_ci_low, mae_ci_high, bias_ci_low, bias_ci_high, rmse_ci_low,
rmse_ci_high, hit1f_ci_low, hit1f_ci_high, model_version, segment_start, period_start, period_end,
computed_at, methodology_version, schema_version`.

Plus `station_id='ALL'` rows aggregating over all stations (mean over stations of daily errors, same bootstrap).

### 3.5 `pairwise` — model-vs-model
Column list: `castcheck.verify.PAIRWISE_COLUMNS`.

`station_id, init_hour, lead_day, variable, window, model_a, model_b, n_common, mae_diff, ci_low,
ci_high, significant(bool), method, computed_at, methodology_version, schema_version`.

`method` is part of the key: a bilinear comparison and a nearest-neighbour comparison are different
comparisons, and mixing them would pair rows that were never computed against each other.

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
- **AIWP**: `https://noaa-oar-mlwp-data.s3.amazonaws.com/{PROD}_{ver}_{INIT}/{YYYY}/{MMDD}/{PROD}_{ver}_{INIT}_{YYYYMMDDHH}_f000_f240_06.nc` (e.g. `GRAP_v100_IFS`); discover `{ver}` by listing the bucket prefix (`?list-type=2&prefix=`). Open with `h5netcdf` over `fsspec` HTTP (block_size 2 MiB — measured faster than 1 MiB and no slower than 4 MiB); variable `t2` shape (41, 721, 1440), chunk = one full layer → read one time index, extract all stations; **f000 is a fill value (9.97e36) → missing**. Longitudes 0..359.75.
  **Version policy**: a product may publish two version directories side by side (FourCastNet ships `v100` and `v200`, overlapping 2020–2023). `AiwpSource` lists the bucket root, and for each initialisation uses the **highest version that actually contains that run**, recording it in `model_version` as `"{PROD}_{ver}"` — so a version change is visible in the data instead of silently altering a series. `AiwpSource(version=...)` pins one version for reproduction. 06/18 UTC initialisations exist for 2023 only and are ignored.
- **NWS CLI**: `https://api.weather.gov/products/types/CLI/locations/{LOC}` (LOC = pil minus `CLI`), then `/products/{id}`; final report = first issuance after local midnight whose text contains the `YESTERDAY` block; parse fixed-width `MAXIMUM  76  355 PM`. Must send a real `User-Agent`. Retention ~7 days; history: `https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil={PIL}&sdate=...&edate=...&fmt=text`.
- **Obs**: `https://api.weather.gov/stations/{ICAO}/observations?start=..&end=..`; `maxTemperatureLast24Hours` is null — derive from hourly `temperature` only as fallback.

## 5. CLI (typer)

```
castcheck [-v] fetch   --model ifs_hres --init 2026-08-30T00   [--stations KNYC,KORD]
castcheck fetch-latest [--lookback-days 3] [--workers 3] [--models gfs,ifs_hres] [--min-retry-h 3]
castcheck backfill gfs 2024-01-01 2024-03-31 [--workers 2] [--no-skip-existing]
castcheck truth    --date 2026-08-29         # CLI first-final for all stations (+ obs fallback)
castcheck truth-backfill 2024-01-01 2024-12-31 [--stations KNYC]   # IEM
castcheck derive                              # values → daily_forecasts
castcheck verify                              # → scores, pairwise
castcheck build-site                          # → public/
castcheck status [--no-fail-on-gaps]          # → public/api/v1/status.json (+ exit 1 if gaps today)
castcheck daily                               # derive → verify → build-site
castcheck publish hf|kaggle|bluesky [--dry-run]   # optional, token-gated
```

All commands are idempotent and safe to re-run. Logging goes to **stderr** (INFO; `-v` for DEBUG;
`CASTCHECK_LOG_JSON=1` for one JSON object per line) so stdout stays parseable; each command ends
with a one-line summary and a `data/raw/last_run.json` entry (§7.1).

**Exit codes.** `status` exits 1 when today has gaps (`--no-fail-on-gaps` to report only).
`fetch-latest` exits 1 only when *every* planned run came back empty — one late upstream file is
normal and must not turn a scheduled workflow red, but a wholly empty pass is a broken pipeline.
Everything else exits 0 unless it raised.

**Skip rules for `fetch-latest`.** A run is planned when (a) `init + availability_delay <= now`
(§7), (b) it is not already stored complete — `store.existing_inits`, which requires `max_h/step_h`
distinct valid times with a present `t2` — and (c) it was not attempted within the last
`--min-retry-h` hours (`store.last_attempt_by_init`). (c) exists because a run upstream never
completes would otherwise be re-downloaded by all four daily passes; the retry surface is bounded
absolutely by `--lookback-days`. The planner is `cli.plan_runs`, a pure function, and is tested
without network or parquet in `tests/test_cli.py`.

**Scoped completeness reads.** `existing_inits` and `last_attempt_by_init` take `start`/`end` and
read only the four columns they need (`init_time, valid_time, variable, missing_reason` /
`init_time, fetched_at`) from only the monthly shards covering that range. Callers that plan work
over a window must pass them; otherwise the whole archive is read on every invocation, which after
a year of backfill is millions of rows per model per pass.

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
| `fetch.yml` | 05:00, 06:00, 08:00, 09:30 (00Z cycle); 18:00, 21:00, 23:00 (12Z cycle) | `fetch-latest` (each run only fetches what is new) |
| `truth.yml` | 10:30, 16:00 | `truth` (yesterday) + `truth --date` two days ago for corrections |
| `publish.yml` | 11:00 | `daily` (= `derive` → `verify` → `build-site`) → `status --no-fail-on-gaps` → Cloudflare Pages deploy → HF/Kaggle/Bluesky (each gated on its secret) → commit data shards |
| `backfill.yml` | manual dispatch | ranges per model |
| `test.yml` | push / PR | `ruff check .` → import `eccodes`/`cfgrib` → `pytest -q -m "not network"` |

**Cron and availability.** A run is fetched once `init + availability_delay` has passed, where the
delay is per source (`cli.AVAILABILITY_DELAY_H`: GFS 5.5 h, ECMWF 8 h) and, for AIWP, per initial
field (GFS-initialised 6 h, IFS-initialised 9.5 h — CIRA has to wait for ECMWF dissemination first).
The 12Z crons are the 00Z crons plus twelve hours; the same delays apply to both cycles. Nothing
depends on hitting a cron exactly: `fetch-latest` looks back three days, so a missed or late run is
picked up by the next pass.

**Concurrency and the commit race.** Every workflow that writes `data/` shares
`concurrency: {group: data-writes}`, so two of ours never run at once. A push that still races
(a human commit, or a run released the instant the group frees) is handled by
`scripts/commit_data.sh`, which retries `push` behind `pull --rebase` up to five times and, because
parquet cannot be merged, resolves a conflict by keeping the shard this run just wrote — safe,
because upserts are idempotent.

**Runner dependencies.** `uv pip install -e .` is enough: `eccodes >= 2.43` depends on `eccodeslib`
on non-Windows platforms, which ships a `manylinux_2_28_x86_64` wheel, so cfgrib finds libeccodes
with no `apt-get`. `test.yml` asserts this with an import step so that a regression in the wheel
chain fails loudly instead of at the next scheduled fetch.

Secrets: `CLOUDFLARE_API_TOKEN` (Pages edit), optional `HF_TOKEN`, `KAGGLE_API_TOKEN`, `BSKY_HANDLE`
/ `BSKY_APP_PASSWORD`. They are injected as **job-level `env`** so that the optional steps can be
skipped with `if: env.HF_TOKEN != ''` — the `secrets` context is not available in a step `if:`, the
`env` context is. Data shards are committed by the workflow with `[data]` commit messages.

### 7.1 Run journal — `data/raw/last_run.json`

Every CLI command appends its outcome to this file (`store.record_run`, atomic write). It is the one
file under `data/raw/` that is committed, because the status page has to show freshness *across*
workflows: the fetch workflow and the publish workflow are different jobs on different runners.

```json
{
  "updated_at": "2026-08-30T11:04:12+00:00",
  "commands": {
    "fetch-latest": {
      "status": "ok",                 // "ok" | "error"
      "started_at":  "2026-08-30T11:02:55+00:00",
      "finished_at": "2026-08-30T11:04:12+00:00",
      "duration_s": 77.4,
      "exit_code": 0,
      "summary": "11/12 run(s) with data",
      "castcheck_version": "0.1.0",
      "last_success_at": "2026-08-30T11:04:12+00:00"
    }
  }
}
```

Contract: `castcheck/cli.py` **writes** it, `castcheck/status.py` **reads** it and nothing else may.
`last_success_at` is sticky — a failing run leaves the previous success timestamp in place, so the
status page can say how stale each stage is. Keys are command names as typed
(`fetch`, `fetch-latest`, `backfill`, `truth`, `truth-backfill`, `derive`, `verify`, `build-site`,
`status`, `publish-hf`, `publish-kaggle`, `publish-bluesky`). A missing or corrupt file means "never
run" and must never be an error.

### 7.2 Local mirror

`scripts/run_daily.sh` (launchd, 07:45 local) runs the same commands, never aborts on the first
failure, and writes `data/raw/last_failure.txt` plus a macOS notification when any step failed; a
clean run deletes that file. `scripts/backfill_local.sh` and `scripts/truth_backfill_local.sh` are
the detached one-model-at-a-time / one-year-at-a-time backfills.

### 7.3 Being polite to upstream

`sources/_http.py` keeps per-host state shared by all threads: a minimum interval between request
starts (`MIN_INTERVAL_S`, default 0.05 s), multiplied by `THROTTLE_GROWTH` on every 429/503 up to
`THROTTLE_MAX_INTERVAL_S` (5 s) and decayed back when the host goes quiet, so one throttling host
slows every worker rather than only the unlucky thread. Retries are counted per host and reported in
one summary line per command (`_http.log_summary`, called at the end of every CLI command) instead
of a warning per attempt. All four knobs are environment variables (`CASTCHECK_HTTP_RETRIES`,
`CASTCHECK_HTTP_BACKOFF_CAP`, `CASTCHECK_HTTP_MIN_INTERVAL`, `CASTCHECK_HTTP_MAX_INTERVAL`).

Index/inventory requests (`.index`, `.idx`) use fewer retries than data requests (`INDEX_RETRIES=3`
vs 6) and share a per-run failure budget (`MAX_INDEX_FAILURES=5`): once a run's indices are clearly
unavailable, the remaining steps become missing rows immediately instead of each burning a full
retry ladder. Before this, one unavailable ECMWF run could occupy a worker for ~20 minutes.

Measured 2026-08-30, two `ifs_hres` runs, one station, `t2` only, while a local backfill was already
hitting the same mirror: 17 requests to `ecmwf-forecasts.s3.amazonaws.com`, **8 answered `503`
(47 %), 8 retries, 0 given up**, 16/16 values present; the host interval rose to the 5 s cap and one
summary line was logged. The portal (`data.ecmwf.int`) served 9 requests with no throttling.

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

## 10. v0.3 — headline redefined after external review (2026-08-31)

External review (meteorologist/statistician, see `docs/06-external-review-v02.md`) found that verifying
*sampled* daily extremes against the *true* NWS extremes makes the headline metric depend on each model's
own diurnal amplitude, and that the shared-resampling bootstrap gives unstable intervals for sparse groups.
v0.3 fixes both. Contracts below are binding for the implementation round; `SCHEMA_VERSION` → `"0.3"`,
`METHODOLOGY_VERSION` → `"0.3"`.

### 10.1 New truth table `truth_instant` (observed 2 m temperature at the common sample instants)
Path `data/truth_instant/year=<YYYY>.parquet`, key `(station_id, valid_time)`.

| column | type | notes |
|---|---|---|
| station_id | str | |
| valid_time | timestamp[UTC] | one of 00/06/12/18 UTC |
| temp_c | float32 | observed 2 m air temperature; NaN if no usable report |
| obs_time | timestamp[UTC] | timestamp of the report actually used |
| source | str | `ASOS_IEM` (IEM ASOS archive, routine METAR, `report_type=3`) or `NWS_API` (api.weather.gov, last 7 days) |
| n_reports | int8 | reports found inside the ±35 min window |
| qc_flag | str | `""`, `"no_report"`, `"gap_gt35min"`, `"suspect"` |
| schema_version, methodology_version | str | |

Rule: use the routine METAR closest to the synoptic hour within ±35 min (prefer the :51–:56 report); ASOS
reports whole °F/tenths °C — store as reported, converted to °C float32. Source module
`castcheck/sources/iem_asos.py` (IEM: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=<ID>&data=tmpf&year1=..&tz=Etc/UTC&format=onlycomma&report_type=3` — IEM ids drop the leading K for CONUS ASOS, e.g. `NYC`, `ORD`; verify per station). Backfill 2024-01-01→ for all stations.

### 10.2 Headline variables (scores.variable)
| variable | forecast | truth | role |
|---|---|---|---|
| `t2` | instantaneous 2 m T at the 4 common instants | `truth_instant.temp_c` at the same instant | **headline** (pooled over the 4 instants) |
| `t2_00z`,`t2_06z`,`t2_12z`,`t2_18z` | same, one valid hour | same | permalink/station pages (diurnal structure) |
| `tmax_s`, `tmin_s` | max/min of the 4 forecast samples | max/min of the 4 **observed** samples (same definition) | like-for-like daily extremes |
| `tmax_cli`, `tmin_cli` | max/min of the 4 forecast samples | NWS CLI daily extremes | secondary "what a daily-max user experiences"; carries the sampling penalty, labelled as such |

Native extremes (`mx2t*`, `TMAX/TMIN`) remain diagnostic columns in `daily_forecasts` and get their own
secondary variables `tmax_native_cli`, `tmin_native_cli` (vs CLI) where available.

### 10.3 Statistics
- **Bootstrap per group** on the group's own realized date axis: circular moving-block, block = 7 days,
  1000 resamples, percentile 95 % CI. CI is `NaN` (site shows "—") when `n < 28` or `n_blocks < 4`.
  Vectorise per group (index matrix `B × n`), not across groups.
- **Proportions** (`hit1f/hit2f/hit3f`): Wilson score 95 % intervals, no bootstrap.
- **Skill**: `skill_persistence = 1 − mae / mae_persistence_common`, with new columns `n_common`,
  `mae_persistence_common`; the persistence baseline for `t2*` is the observation at the same UTC hour
  `lead_day` days earlier; for `tmax_s/tmin_s/tmax_cli/tmin_cli` the observed extreme `lead_day` days earlier.
  `skill_ci_low/high` via the same paired bootstrap.
- **Debiased skill** becomes out-of-sample: bias estimated on the trailing 30 scored days *before* each
  day (min 15 days), applied forward; days without enough history are excluded from `mae_debiased`
  (`n_debiased` column). Never in-sample.
- **Multiple comparisons**: pairwise keeps `mae_diff, ci_low, ci_high`; rename `significant` →
  `distinguishable_uncorrected`; add `p_boot` (two-sided bootstrap p) and `distinguishable_holm`
  (Holm over the family of comparisons against the leader within one displayed table: same station,
  init, lead, variable, method, window). The site marks ▼/▲ only on `distinguishable_holm`.
- **ALL row** unchanged (per-day cross-station mean of each functional separately) — wording fixed in §4.

### 10.4 Station metadata
`config/stations.yaml` gains `grid_elev_m` (mean elevation of the 0.25° cell containing the station,
from the public-domain ETOPO 2022 60-arc-second grid, computed once by `scripts/build_stations.py
--grid-elev`) and derived `dz_m = elev_m − grid_elev_m`; `/stations/` and stations.csv show `dz_m` and
the first-order lapse-rate magnitude `|dz_m| × 6.5 K/km`. The `kalshi` column is renamed
`market_city` and the selection rule is stated in METHODOLOGY §1.

### 10.5 Site/doc corrections from the review
Fix code URL on /data/ (github.com/ZzZTripleZzZ/castcheck); changelog v0.2/v0.3 on /data/; §10 title;
/models/ reference to §7; /stations/ "scored days" must exclude baselines; status uptime counted from each
model's `period_start`; map shows bias of a fixed reference (IFS HRES) and an all-model mean, never the
per-station winner; add `CITATION.cff`, source commit hash in the footer, `forecast_values` download on
/data/; `mae` described without a sign clause; lead-hour/longitude note in §2.5.

### 10.6 Phenomenon note (cautious)
The observed larger cold bias of the AI models at 18/00 UTC is reported as an *unattributed observation*;
the ERA5-training explanation is listed as a hypothesis alongside sampling and initial-condition effects,
with the planned attribution analyses (per-hour bias, native extremes, ERA5 run through the same pipeline,
seasonal split). No causal wording on the site until those are done.
