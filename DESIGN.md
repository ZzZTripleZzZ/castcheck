# CastCheck — Engineering Design (methodology v0.3.1, schema v0.3)

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
    iem_asos.py            IEM ASOS archive: routine METARs for truth_instant (§10.1)
  derive.py                station values → sampled/native daily extremes per (model, init, station, lead_day)
  truth.py                 assemble truth_daily from CLI/CF6/obs with first-final policy and qc flags
  truth_instant.py         truth_instant: the METAR nearest each synoptic hour, ±35 min (§10.1)
  merge.py                 shared join helpers for forecast values × truth
  verify.py                scores, bootstrap CIs, pairwise comparisons, persistence baseline
  site/
    build.py               static site generator (Jinja2) → public/
    templates/*.html
    svg.py                 inline SVG charts and maps drawn at build time
    assets/                css, small js (charts drawn client-side from JSON; no build step)
  api.py                   JSON export → public/api/v1/...
  schedule.py              availability deadlines shared by cli.plan_runs and status.build
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
  release_local.sh         build + deploy the site from a laptop when Actions is unavailable
  clean_conflict_copies.sh delete the "name 2.parquet" copies a syncing filesystem leaves behind
  health_gaps.py           gap report over the archive, independent of status.py
  crosscheck_grid.py, crosscheck_verify.py, crosscheck_bootstrap.py   independent recomputations (§8)
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
    lat: float | None; lon: float | None; elev_m: float | None
    market_city: str | None   # informational; the `kalshi` field of v0.2 (§10.4)
    iem_id: str | None        # IEM ASOS archive id, frozen per station (§10.1)
    grid_elev_m: float | None # mean elevation of the 0.25° cell (§10.4); dz_m is a property

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

All tables carry `schema_version: str` (currently `"0.3"`, `castcheck.SCHEMA_VERSION`) and
`methodology_version: str` (currently `"0.3.1"`, `castcheck.METHODOLOGY_VERSION`).

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

The authoritative column list is `castcheck.store.DAILY_COLUMNS`; this table must be edited in the
same change as that constant.

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
| tmax_obs_s_c, tmin_obs_s_c | float32 — the observed max/min of the *same* four instants, the truth for `tmax_s`/`tmin_s` (v0.3, METHODOLOGY §2.3) |
| n_obs_samples | Int8 (0–4) — how many of the four observations were present; a day is scored like for like only at 4 |
| native_overhang_h | Int8, hours — the overhang the native bucket run actually realised (METHODOLOGY §2.4), 0 when there is no native value |
| schema_version, methodology_version | str |

### 3.4 `scores` — published aggregates
Path: `data/scores/latest.parquet` (overwritten daily), `data/scores/pairwise_latest.parquet`, and a
dated snapshot `data/scores/history/<YYYY-MM-DD>.parquet`.

**History retention.** A snapshot is a few MB and one is written every day, so an unpruned history
grows by roughly a gigabyte a decade inside a git repository that every workflow run has to clone.
`castcheck prune-history` (run by `publish.yml` immediately after `daily`, in the same commit that
adds the day's snapshot) keeps **every snapshot from the last 90 days** and **the 1st of every month
forever**, and deletes the rest. That is dense enough near the present to bisect a regression day by
day, and dense enough far from it to plot a decade of monthly milestones. Files whose name is not a
plain `YYYY-MM-DD.parquet` — conflict copies, anything a human put there — are never touched. The
retention window is `cli.HISTORY_KEEP_DAYS`; the rule itself is the pure `cli._history_prune_plan`,
which is what the tests exercise against a fixed date.

The authoritative column list is `castcheck.verify.SCORE_COLUMNS`; this table must be edited in the
same change as that constant.

`station_id, model_id, init_hour, lead_day, variable, method, window(30d|90d|365d|all),
n, n_stations, n_flagged, mae, bias, rmse, hit1f, hit2f, hit3f, mae_debiased, n_debiased,
n_common, mae_persistence_common, skill_persistence, skill_persistence_debiased,
skill_ci_low, skill_ci_high, mae_ci_low, mae_ci_high, bias_ci_low, bias_ci_high, rmse_ci_low,
rmse_ci_high, hit1f_ci_low, hit1f_ci_high, model_version, segment_start, period_start, period_end,
computed_at, methodology_version, schema_version`.

`variable` is one of `castcheck.verify.VARIABLES` (§10.2): `t2`, `t2_00z`, `t2_06z`, `t2_12z`,
`t2_18z`, `tmax_s`, `tmin_s`, `tmax_cli`, `tmin_cli`, `tmax_native_cli`, `tmin_native_cli`. The v0.2
names `tmax`/`tmin` no longer exist — the closest equivalent is `tmax_cli`/`tmin_cli`, and the
headline is `t2`.

v0.3 columns and their meaning:

| column | meaning |
|---|---|
| `n_debiased` | scored days that had ≥ 15 of the previous 30 scored days available, i.e. the days `mae_debiased` is computed over. `mae_debiased` is **out of sample**: the bias is estimated on the trailing days *before* each day and applied forward. |
| `n_common` | days on which both this model and the persistence baseline have a value — the denominator `skill_persistence` is actually computed on. 0 for the baseline's own rows. |
| `mae_persistence_common` | the baseline MAE over exactly those `n_common` days. `skill_persistence == 1 − mae_over_common / mae_persistence_common`; the site must show this number next to the skill column instead of the baseline row's own `mae` (review item A1). |
| `skill_ci_low/high` | percentile CI of `skill_persistence` from the paired bootstrap on the common days. |
| `hit1f_ci_low/high` | **Wilson** score interval (not bootstrap), so a 0 % hit rate reports `[0, 0.12]` rather than `[0, 0]`. |

`mae_ci_*`, `bias_ci_*`, `rmse_ci_*`, `skill_ci_*` are `NaN` when the group has `n < 28` days or
fewer than 4 blocks (§10.3); the site shows "—".

Plus `station_id='ALL'` rows aggregating over all stations (per-day cross-station mean of each
functional separately, then the same statistics and bootstrap over days).

### 3.5 `pairwise` — model-vs-model
Column list: `castcheck.verify.PAIRWISE_COLUMNS`.

`station_id, init_hour, lead_day, variable, window, model_a, model_b, n_common, mae_diff, ci_low,
ci_high, p_boot, distinguishable_uncorrected(bool), distinguishable_holm(bool), method,
computed_at, methodology_version, schema_version`.

`method` is part of the key: a bilinear comparison and a nearest-neighbour comparison are different
comparisons, and mixing them would pair rows that were never computed against each other.

`significant` was renamed `distinguishable_uncorrected` in v0.3 (review item B2) and must not be used
to mark ▼/▲ anywhere on the site. `p_boot` is the two-sided bootstrap p-value of the paired MAE
difference; `distinguishable_holm` is the Holm-corrected verdict over the family of comparisons
**against the leading model within one displayed table** (same `station_id, init_hour, lead_day,
variable, method, window`). Pairs outside that family are `False` by construction, so the site marks
▼/▲ only on the leader column.

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
def daily_from_values(values, stations, models, truth_instant=None) -> pd.DataFrame  # daily_forecasts
def derive_window(start_date, end_date, stations=None, models=None, ...) -> pd.DataFrame  # incremental, by *init* date
def instant_errors(values, truth_instant, stations=None, models=None) -> pd.DataFrame     # §10.2
def observed_sampled_extremes(truth_instant, stations=None) -> pd.DataFrame               # obs max/min of the 4 instants
# truth.py
def assemble_truth(cli_rows, cf6_rows, obs_rows) -> pd.DataFrame                # truth_daily with first-final + qc
# verify.py
def score(daily, truth, instant=None, windows=(30,90,365,None), n_boot=1000, seed=0, truth_instant=None) -> tuple[pd.DataFrame, pd.DataFrame]  # scores, pairwise
def persistence_daily(truth: pd.DataFrame) -> pd.DataFrame     # legacy CLI-truth baseline; `score` builds its own per variable
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
castcheck truth-backfill 2024-01-01 2024-12-31 [--stations KNYC]   # IEM AFOS
castcheck truth-instant --date 2026-08-29     # observed 00/06/12/18 UTC, api.weather.gov (recent days)
castcheck truth-instant-backfill 2024-01-01 2024-12-31             # the same, from the IEM ASOS archive
castcheck truth-qc                            # re-check stored CLI extremes against truth_instant (§3.3)
castcheck derive                              # values → daily_forecasts
castcheck verify                              # → scores, pairwise
castcheck build-site                          # → public/
castcheck status [--no-fail-on-gaps]          # → public/api/v1/status.json (+ exit 1 if *due* gaps today)
castcheck prune-history [--keep-days N]       # thin data/scores/history/ (§3.4)
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
- **Weight budget.** `public/` must stay under 250 MB and 20 000 files (the Cloudflare Pages deploy limit), with no file above 25 MiB. The generator reports both at the end of `build-site`. What keeps it there: `castcheck.site.build.minify_html` strips the template indentation; the tables use short cell classes (`n`/`c`/`s`/`k`/`p`/`bb`) with the value in the cell's own text rather than a wrapper `span`; two windows covering the same days share one row on a permanent link; `/data/` carries only the small tables plus the per-station `daily_errors/{ICAO}.csv.gz`, and the bulk pairwise table is on Hugging Face.
- `/api/v1/scores/index.json` (alias `latest.json`), `/station/{ICAO}/cards.json`, `/api/v1/scores/leaderboard.json`, `/api/v1/leaderboard/{view}.json`, `/api/v1/pairwise/latest.json`, `/api/v1/stations.json`, `/api/v1/models.json`, `/api/v1/status.json` — plain static JSON.
  `/station/{ICAO}/cards.json` is the per-station bundle: the station's whole `scores` table plus one entry in `cards` per model and lead day (its pairwise slice and daily error series). It replaces the ~1 800 `/api/v1/scores/{station}/{model}/{lead}.json` files, which each repeated the envelope and a slice of a table that was published in full a second time under `scores/by-station/`; that path is now a pointer document. Inside the bundle the low-cardinality label columns (`model_id`, `variable`, `method`, `window`, `model_version`, `segment_start`, `period_start`, `period_end`) are dictionary-encoded: the row holds an index into `scores.dictionaries[column]`. Every other endpoint stays plain `{columns, rows}`.

Charts: minimal inline SVG or a tiny client-side script reading the JSON; no framework, no build step. Pages must be readable with JS disabled (tables first, charts as enhancement). Theme: light, system font, one accent colour.

## 7. Pipelines (GitHub Actions; local launchd mirrors the same commands)

| workflow | schedule (UTC) | steps |
|---|---|---|
| `fetch.yml` (daily) | 05:00, 06:00, 08:00, 09:30 (00Z cycle); 18:00, 21:00, 23:00 (12Z cycle) | `fetch-latest --workers 3` (each run only fetches what is new) |
| `fetch.yml` (sweep) | **03:00 Sun** | `fetch-latest --lookback-days 10 --min-retry-h 0` — one weekly pass that re-asks upstream for everything still incomplete in ten days |
| `truth.yml` | 10:30, 16:00 | `truth` (yesterday) + `truth --date` two days ago + `truth-instant` + `truth-instant-backfill` (10 d) + `truth-qc --start` (15 d) |
| `publish.yml` | 11:00 | `daily` (= `derive` → `verify` → `build-site`) → `status --no-fail-on-gaps` → `prune-history` → Cloudflare Pages deploy → HF/Kaggle/Bluesky (each gated on its secret) → commit data shards |
| `health.yml` | **12:30** | `status --no-fail-on-gaps` → `scripts/health_gaps.py`; opens/updates/closes one `data-gap` issue (§7.4) |
| `links.yml` | **09:00 Mon** | lychee over the deployed site plus the outbound links in `README.md` and `METHODOLOGY.md`; opens/closes a `link-rot` issue |
| `deps.yml` | **04:00 on the 1st** | `uv lock --upgrade` → ruff + eccodes import + full test suite → opens a **pull request** if the lock changed (never auto-merged) |
| `consistency.yml` | **13:00 on the 2nd** | `derive --full` + `verify` on the pinned snapshot → `crosscheck_verify.py --compare-incremental`; opens/closes a `consistency` issue (§7.5) |
| `backfill.yml` | manual dispatch | ranges per model |
| `test.yml` | push / PR | `ruff check .` → import `eccodes`/`cfgrib` → `pytest -q -m "not network"` |
| `_failure-issue.yml` | called, never scheduled | the shared `if: failure()` handler every workflow above ends with (§7.4) |

Every time in that table is UTC and every one is deliberate: `health.yml` is after `publish.yml` has
committed the day's data, `consistency.yml` is on the 2nd so a `publish.yml` run on both the 1st and
the 2nd has already written the scores it compares against, and `deps.yml` is at 04:00 — the only
hour of the day with no fetch, truth or publish job in it.

**Cron and availability.** A run is fetched once `init + availability_delay` has passed, where the
delay is per source (GFS 5.5 h, ECMWF 8 h) and, for AIWP, per initial field (GFS-initialised 6 h,
IFS-initialised 9.5 h — CIRA has to wait for ECMWF dissemination first). Those constants live in
**`castcheck/schedule.py`**, which `cli.plan_runs` and `status.build` both import: the fetcher's
"not worth asking for yet" and the status page's "not a gap yet" must be the same judgement, or the
page shows a red bar for a run nobody has published. The same module holds the CLI truth deadline
(the station's own local midnight plus 1–4 h, so at 06 UTC yesterday's report exists in New York
and cannot exist in Los Angeles) and the observed-instant deadline (18 UTC + 1.5 h). Days on the
wrong side of a deadline are drawn grey (`not_due_yet`) and are in neither the gap list nor the
uptime denominator.
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

### 7.4 Alerting — nobody is watching the Actions tab

A pipeline meant to run for ten years unattended needs its failures to arrive somewhere a person
actually looks. Two mechanisms, both writing to GitHub issues, both self-closing:

**Workflow failure → `pipeline-failure` issue.** Every workflow ends with a job that calls the shared
reusable workflow `.github/workflows/_failure-issue.yml` under `if: failure()`. Deduplication is by
*(workflow, UTC day)*: the first failure of a workflow on a day opens
`[pipeline-failure] <workflow> failed on <date>`, and every later failure of the same workflow that
day is appended as a **comment**. A day of flapping is then one thread instead of twenty issues, and
the issue list stays a list of bad days. The handler does no `actions/checkout` — it has to work on
the run where the checkout is what failed — so its script is inline and its only tool is the runner's
`gh`. `test.yml` calls it only for pushes to `main`; a red PR check is already in front of the person
who caused it, and an issue per red PR would train people to ignore the label.

Because a reusable workflow may never request more permission than its caller, and every data
workflow sets `permissions: {contents: write}` at the workflow level, each calling job overrides it
with `permissions: {contents: read, issues: write}`.

**Data overdue → `data-gap` issue.** `health.yml` (12:30 UTC daily) recomputes `status.json` from the
committed `data/` — not from the deployed site, so it still works on a day when the Cloudflare deploy
is what broke — and hands it to `scripts/health_gaps.py`. That script asks the sharper question
`castcheck status` cannot: not "is anything missing right now" (at 12:30 a 12Z run is legitimately
absent, and one failed fetch is picked up an hour later) but **"has anything been overdue for more
than 24 h"**. It reads the per-day grids rather than `current_gaps`, because `current_gaps` only
covers today while a slot stuck since Tuesday is exactly what this looks for; each grid row already
carries the `due_at` that `castcheck/schedule.py` computed, so "how long overdue" is read off the data
and no state has to be carried between runs. Exit code 2 opens or updates a single `data-gap` issue,
exit code 0 closes it.

One deliberate narrowing: only slots whose deadline passed within the last **7 days** can raise the
alarm. An in-flight backfill leaves months of legitimately empty slots behind it, and an alarm that
counted them would be red the day it was switched on and stay red forever — which is the same as
having no alarm. The older holes are still counted in the issue body, so the number never silently
disappears.

### 7.5 Self-checks — catching what does not fail loudly

`deps.yml` (04:00 on the 1st) runs `uv lock --upgrade`, then ruff, the eccodes/cfgrib import check and
the full test suite against the new lock, and opens a **pull request** with a table of the version
changes. It never merges. A dependency bump that silently moves a published number is precisely the
class of error this project exists to make visible, so a human reads the diff; the PR body says to
dispatch `consistency.yml` on the branch if numpy, pandas, pyarrow, xarray or eccodes moved. (Needs
"Allow GitHub Actions to create and approve pull requests" enabled in the repository settings.)

`links.yml` (09:00 Monday) walks the deployed site's key routes and every outbound link in `README.md`
and `METHODOLOGY.md` with lychee. A verification project's credibility rests on its citations
resolving, and over ten years the NWS reorganises its product pages and publishers move their DOIs.
The rules live in `lychee.toml` rather than in the workflow's args, so they are reviewable in a diff:
excluded there are Kalshi's market pages (Cloudflare-protected and geofenced — a 403 from every
runner, a working link for a reader) and the login-walled hosts, which answer a signed-out GET with a
sign-in page and say nothing about whether the target exists.

`consistency.yml` (13:00 on the 2nd) is the one that guards the numbers themselves. The daily pipeline
is incremental by necessity — `derive --since 14` reopens two weeks of initialisations and upserts
them, which is what makes a ten-year pipeline affordable — and that means a bug in the upsert path, a
dtype that rounds, or a shard written by an older code version would drift the published scores a
little at a time, invisibly. So once a month the whole archive is re-derived from `forecast_values`,
re-scored, and required to reproduce the incremental answer to **1e-6** on `n`, `mae`, `bias`, `rmse`,
the hit rates, `mae_debiased` and `skill_persistence` (`scripts/crosscheck_verify.py
--compare-incremental`). A row present on only one side is a failure too, not something to skip.

The comparison is only meaningful if both sides see the same tables, so the workflow first pins
`data/` to the commit that last wrote `data/scores/latest.parquet` (`git checkout <sha> -- data/`,
code stays at HEAD). `verify` takes its `as_of` — and therefore the 30d/90d/365d window edges — from
`max(climo_date)` in the data rather than from the clock, so with the same snapshot in front of them
the two paths must agree exactly. Without the pin, forecasts that arrived between the 11:00 publish
and 13:00 would land in the full recompute only, and every window would "disagree" for a reason that
is not a bug.

`health.yml`, `links.yml` and `consistency.yml` all stay **out of** the `data-writes` concurrency
group and hold `contents: read`. None of them commits, and a monitor that could be blocked by the very
job it is monitoring is not a monitor.

### 7.6 Hugging Face history — the annual squash

`publish/hf.py` pushes one commit a day, and because every push rewrites the same parquet shards, a
year of pushes is also a year of *superseded* blobs. The Hub keeps every blob forever, so an
un-squashed decade would make `git clone` of the dataset download ten years of dead revisions to
reconstruct one day's table. `maybe_squash()` therefore calls `super_squash_history` once a year, on
1 January, before that day's upload.

This is **irreversible and non-fast-forward**: every past commit SHA disappears, and anyone who pinned
`revision="<sha>"` gets a 404 afterwards. That is the deliberate trade, and it is why CastCheck's
citable snapshots are the dated files in `data/scores/history/` and the tagged releases of the *code*
repo — never Hub commit SHAs. Two safety properties: it is idempotent (after a squash the repo has one
commit, so the `> 1` test is false for every later push that day), and a squash that fails or cannot
be listed returns a note and lets the upload proceed, because housekeeping must never cost a day of
data. `CASTCHECK_HF_SQUASH=1` forces an attempt on any date, `=0` disables it for anyone mirroring to
a repo whose history they want kept.

### 7.7 Timeouts

`timeout-minutes` is set from measured cost with room for a growing archive, not guessed. From
`data/raw/last_run.json` on 2026-09-01: `fetch-latest` 300 s for 22 runs, `derive --full` 103 s,
`verify` 586 s, `build-site` 1312 s, `daily` 2002 s.

| workflow | timeout | basis |
|---|---|---|
| `fetch.yml` (daily) | 50 min | ~5 min measured; the headroom is for the retry ladders when a host is throttling (§7.3 measured 47 % `503` from one ECMWF mirror) |
| `fetch.yml` (sweep) | 120 min | ~5x the daily pass: ten days of slots, and `--min-retry-h 0` means it cannot skip what it already tried this morning |
| `truth.yml` | 20 min | measured ~70 s for all five steps; text products, not grids |
| `publish.yml` | 90 min | `daily` alone is ~33 min and grows with the archive — the previous 40 min left no room. 90 is ~2.5x current cost and far below the 6 h job ceiling |
| `health.yml` | 30 min | one `status` build over the completeness window |
| `consistency.yml` | 180 min | ~7x the measured `derive --full` + `verify` + comparison; the right multiple for something that runs monthly on an archive that grows all year |
| `deps.yml` | 45 min | a cold `uv sync` plus the full test suite |
| `links.yml` | 20 min | ~30 URLs at concurrency 4 |
| `_failure-issue.yml` | 10 min | two `gh` calls |

Every third-party action is pinned to a **40-character commit SHA** with the human-readable tag in a
trailing comment (`actions/checkout`, `astral-sh/setup-uv`, `cloudflare/wrangler-action`,
`lycheeverse/lychee-action`). `actionlint` — with `shellcheck` enabled, which is where the inline
`run:` scripts get checked — is clean across all ten workflow files.

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
`METHODOLOGY_VERSION` → `"0.3"`, since raised to `"0.3.1"` by the CLI plausibility check
(METHODOLOGY §3.3), which changed published values but not this schema.

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
