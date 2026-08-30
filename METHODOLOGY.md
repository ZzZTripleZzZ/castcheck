# CastCheck Methodology

**Version 0.1 — 2026-08-30 (pre-release).** This document is versioned together with the data. Any change that alters a published score bumps the version and is listed in the changelog at the end.

CastCheck is an independent, automated, station-level verification of publicly available weather forecasts. It answers one narrow question, every day, with full history: *at this station, for this lead time, how far was each model's raw 2 m temperature forecast from what the National Weather Service later reported?*

## 1. Scope (v0.1)

| Dimension | v0.1 |
|---|---|
| Variable | 2 m air temperature: daily maximum and minimum (and the underlying 6-hourly instantaneous values) |
| Stations | 23 U.S. airport / first-order stations with an NWS daily climate report (CLI); see `config/stations.yaml` |
| Models | ECMWF IFS HRES, ECMWF AIFS Single, NCEP GFS, and the NOAA/CIRA AIWP operational runs of GraphCast, Pangu-Weather, FourCastNet v2 and Aurora (each from both GFS and IFS initial conditions) |
| Initializations | 00 UTC and 12 UTC, scored separately |
| Lead | Lead day 0–9 (all computed); the headline pages show lead days 1, 3, 5 and 7 |
| Truth | NWS Daily Climate Report (CLI), first final issuance |
| Baseline | Persistence (yesterday's observed value) |

Everything else — precipitation, wind, ensembles/probabilities, non-U.S. stations, post-processed products — is out of scope for v0.1 and will be added only after the temperature line has run for 90 days and survived public review.

## 2. Definitions

### 2.1 Climatological day
A **climatological day** is midnight-to-midnight in **local standard time (LST)**, exactly as the NWS CLI product defines it. Daylight-saving time is *not* applied: during DST the climatological day runs from 01:00 to 01:00 local clock time. Each station carries a fixed standard UTC offset (e.g. New York −5 h all year). A day is the half-open UTC interval `[day_start, day_end)`.

### 2.2 Common samples
For every model run and station we take the **instantaneous 2 m temperature at forecast valid times of 00, 06, 12 and 18 UTC**. All models in scope provide these steps. Models with finer output (IFS at 3-hourly steps out to 144 h) are deliberately *not* sampled more finely for the headline scores, so that every model is evaluated on an identical sampling.

For any U.S. station, each climatological day contains exactly four common samples (e.g. for a −5 h station the day `[05Z, 05Z+24h)` contains 06Z, 12Z, 18Z and 00Z of the following UTC date).

### 2.3 Sampled daily extremes (headline)
```
Tmax_sampled(day) = max of the four common samples in the day
Tmin_sampled(day) = min of the four common samples in the day
```
**This under-samples the true diurnal cycle: the sampled maximum is systematically lower than the true afternoon peak and the sampled minimum systematically higher than the true pre-dawn trough.** The bias applies identically to every model and is therefore fair for *comparing* models; it also means the absolute errors reported here are *not* the error a user of a post-processed daily forecast would experience. See §7.

### 2.4 Native extremes (diagnostic only)
Some models output time-window extremes directly: IFS `mx2t3/mn2t3` (≤144 h) and `mx2t6/mn2t6` (>144 h), GFS `TMAX/TMIN` over 6-hour buckets. We compute daily native extremes as the max/min over all buckets lying entirely inside the climatological day, and publish them in a separate diagnostic column. **Native extremes are never mixed into the headline ranking**, because the AI models have no equivalent field and would be penalised purely for a missing output type.

### 2.5 Lead day
`lead_day = target climatological date − UTC date of the model initialization`. A 00 UTC run on 30 August scored against the 31 August climatological day is lead day 1. Because 00 UTC and 12 UTC runs reach the same target day at different forecast hours, the two initializations are always scored and displayed separately.

### 2.6 Station value
The model value at a station is obtained by **bilinear interpolation from the four surrounding grid nodes** of the model's native 0.25° latitude–longitude grid (primary). The **nearest-node** value is computed and published alongside as a sensitivity variant. No elevation or lapse-rate adjustment is applied in v0.1; station elevation and the model grid-cell elevation are listed in the station metadata so that readers can judge representativeness. A lapse-rate-corrected variant is planned as a further sensitivity column, never as the headline.

## 3. Truth

The truth for each station-day is the daily maximum and minimum temperature in the **NWS Daily Climate Report (AFOS product `CLIxxx`)**, taken from the `YESTERDAY` block of the **first CLI issued after local midnight** — the final report for the day. Intermediate same-day reports (`TODAY ... VALID AS OF ...`) are ignored. Later corrected reports are stored with their issuance time in a `revised` column but **do not change published scores** ("first-final" policy). The NWS Preliminary Monthly Climate Data (CF6) is used for month-end reconciliation and to fill gaps; hourly station observations from `api.weather.gov` are used only as a flagged fallback and for quality control (hourly sampling misses peaks by ~1 °F).

NWS reports whole degrees Fahrenheit; forecasts are stored in °C at float precision. Errors are computed in °C and displayed in °F.

## 4. Scores

For each `station × model × initialization × lead_day × variable` and for the windows *last 30 days, last 90 days, last 365 days, all available*, we publish:

| Score | Definition |
|---|---|
| n | number of scored days |
| MAE | mean absolute error, forecast − observed |
| Bias (ME) | mean error, forecast − observed (positive = too warm) |
| RMSE | root mean square error |
| Hit rate ±1 / ±2 / ±3 °F | fraction of days with absolute error within the threshold |
| Skill vs persistence | `1 − MAE_model / MAE_persistence`, computed on the same days |

Windows with n < 30 are shown greyed and excluded from rankings. Scores are computed only within each model's own data-availability period; every page shows the availability bar per model.

## 5. Uncertainty

Every score carries a **95 % bootstrap confidence interval** (1000 resamples of scored days with replacement). **Model-vs-model comparisons use a paired bootstrap on the day-by-day difference series** over the days on which both models have a valid forecast; two models are reported as distinguishable only when the CI of the MAE difference excludes zero. Overlapping single-model intervals are not evidence of equivalence.

## 6. Missing data and quality control

- If any of the four common samples is unavailable for a model run, that station-day is missing for that model and is recorded as an explicit row with a `missing_reason`; nothing is silently skipped.
- Per-model headline scores use all of that model's valid days (n is shown). Pairwise comparisons use the intersection.
- If the CLI value and the hourly-observation-derived extreme differ by more than 2 °F the day is flagged (`qc_flag`) and reported with the flag; flagged days remain in the scores, and the count of flagged days is shown.
- Missing (`M`) or trace values in CLI make the day missing for that variable.

## 7. Fairness statement

These are **raw model outputs on the native 0.25° grid, without MOS, bias correction, downscaling or any post-processing**. They are not equivalent to the products end users receive from a weather service or app, and the scores here **understate operational forecast quality**. Model runs initialized from different analyses (AIWP models from GFS vs IFS initial conditions) are listed as separate models. Model version identifiers are stored with every value; when a model changes cycle or weights, the change is recorded as a segment boundary and scores are not aggregated across it.

## 8. Data availability and backfill

| Source | Available from | Notes |
|---|---|---|
| ECMWF IFS HRES (Open Data, AWS mirror `ecmwf-forecasts`) | 2023-01-18 | CC-BY-4.0 |
| ECMWF AIFS Single | 2025-03 | CC-BY-4.0 |
| NCEP GFS 0.25° (AWS `noaa-gfs-bdp-pds`) | 2021-01-01 | public domain |
| AIWP GraphCast / Pangu / FourCastNet / Aurora (AWS `noaa-oar-mlwp-data`) | 2020–2022 depending on model and initial field | NOAA open data |
| NWS CLI (via api.weather.gov; history via Iowa Environmental Mesonet AFOS archive) | multi-year | public domain |

History is backfilled from these archives; each model is scored only over its own available period.

## 9. Versioning

- `methodology_version` (this document), `schema_version` (data layout) and `model_version` (per value) are stored in every published table.
- Scores are recomputed from the archived station values whenever the methodology version changes; earlier versions remain downloadable.

## 10. Known limitations (v0.1)

1. Four samples per day understate diurnal extremes (§2.3).
2. No elevation adjustment; mountain and coastal stations may show representativeness error that is not model error.
3. Truth is a single station observation with its own uncertainty (~0.5 °F rounding, occasional sensor issues).
4. Only two initializations and one variable.
5. Twenty-three stations in one country.

## Changelog

- **0.1 (2026-08-30)** — initial pre-release methodology.
