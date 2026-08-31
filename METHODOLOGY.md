# CastCheck Methodology

**Version 0.2 — 2026-08-30 (pre-release).** This document is versioned together with the data. Any change that alters a published score bumps the version and is listed in the changelog at the end.

CastCheck is an independent, automated, station-level verification of publicly available weather forecasts. It answers one narrow question, every day, with full history: *at this station, for this lead time, how far was each model's raw 2 m temperature forecast from what the National Weather Service later reported?*

## 1. Scope (v0.2)

| Dimension | v0.2 |
|---|---|
| Variable | 2 m air temperature: daily maximum and minimum (and the underlying 6-hourly instantaneous values) |
| Stations | 23 U.S. airport / first-order stations with an NWS daily climate report (CLI); see `config/stations.yaml` |
| Models | ECMWF IFS HRES, ECMWF AIFS Single, NCEP GFS, and the NOAA/CIRA AIWP operational runs of GraphCast, Pangu-Weather, FourCastNet v2 and Aurora (each from both GFS and IFS initial conditions) |
| Initializations | 00 UTC and 12 UTC, scored separately |
| Lead | Lead day 0–9 (all computed); the headline pages show lead days 1, 3, 5 and 7 |
| Truth | NWS Daily Climate Report (CLI), first final issuance |
| Baseline | Lagged persistence (the observation of day `D − lead`) |

Everything else — precipitation, wind, ensembles/probabilities, non-U.S. stations, post-processed products — is out of scope for now and will be added only after the temperature line has run for 90 days and survived public review.

## 2. Definitions

### 2.1 Climatological day
A **climatological day** is midnight-to-midnight in **local standard time (LST)**, exactly as the NWS CLI product defines it. Daylight-saving time is *not* applied: during DST the climatological day runs from 01:00 to 01:00 local clock time. Each station carries a fixed standard UTC offset (e.g. New York −5 h all year). A day is the half-open UTC interval `[day_start, day_end)`.

This matches the instrument: the ASOS Daily Summary Message, from which the CLI daily maximum and minimum are taken, "runs from 00:00 to 23:59 LST" ([ASOS User's Guide](https://www.weather.gov/media/asos/aum-toc.pdf), §3.1.3). Consequently the DST changeover dates are ordinary 24-hour days here, and Phoenix (−7 h, never on DST) is handled by exactly the same arithmetic as Denver.

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
Some models output time-window extremes directly: IFS `mx2t3/mn2t3` (≤144 h) and `mx2t6/mn2t6` (>144 h), GFS `TMAX/TMIN` over 6-hour buckets. These buckets are anchored to whole multiples of the bucket length from 00 UTC, so they align with a climatological day only when the station's standard offset is a multiple of the bucket length — in practice only the −6 h (CST) stations. Requiring the buckets to lie *entirely* inside the day (the v0.1 rule) therefore produced a value at 8 of 23 stations and `NULL` at the other 15, which is not a usable diagnostic.

**v0.2 rule.** The day is covered by the **contiguous run of buckets that overlaps it**: every bucket inside the day, plus at most one crossing bucket at each end. The run must be gap-free and non-overlapping, must cover the whole day, and may overhang it by at most 6 h in total. The overhang is a deterministic property of the station and the bucket length, not of the date:

| station standard offset | 3 h buckets (IFS ≤144 h) | 6 h buckets (IFS >144 h, GFS) |
|---|---|---|
| −6 h (CST) | 0 h — exact tiling | 0 h — exact tiling |
| −5 h, −7 h, −8 h | 3 h (2 h before + 1 h after, or 1 h + 2 h) | 6 h |

`castcheck.derive.native_overhang_hours(std_offset_h, bucket_h)` returns it. A native maximum that is contaminated by an overhang can only be *too extreme*, never too flat, and the contamination is the same for every model at a given station, so it does not distort a model-vs-model comparison at that station. It does make the native column slightly less comparable *across* stations, which is one more reason it is a diagnostic. The day that straddles the IFS 144 h change of bucket length is accepted if the 3 h and 6 h buckets happen to form one gap-free run, and is `NULL` otherwise.

**Native extremes are never mixed into the headline ranking**, because the AI models have no equivalent field and would be penalised purely for a missing output type.

### 2.5 Lead day
`lead_day = target climatological date − UTC date of the model initialization`. A 00 UTC run on 30 August scored against the 31 August climatological day is lead day 1. Because 00 UTC and 12 UTC runs reach the same target day at different forecast hours, the two initializations are always scored and displayed separately.

A day is scored only when **all four** of its common samples fall inside `(init, init + max_h]`. Two consequences are visible in the tables and are not errors: a 12 UTC run never produces lead day 0 for a U.S. station (its 06 UTC sample precedes the run), and at a 240 h horizon a 00 UTC run reaches lead day 9 only for −5 h and −6 h stations — the −7 h and −8 h stations stop at lead day 8, so the `ALL` row at lead day 9 rests on fewer stations. The number of stations behind every `ALL` row is published as `n_stations`.

### 2.6 Station value
The model value at a station is obtained by **bilinear interpolation from the four surrounding grid nodes** of the model's native 0.25° latitude–longitude grid (primary). The **nearest-node** value is computed and published alongside as a sensitivity variant. Sources use two different conventions — ECMWF stores longitude −180…179.75, GFS and AIWP store 0…359.75, and all three store latitude descending — and the extraction normalises both to give bit-identical answers (verified to 1×10⁻¹³ K on a synthetic field at all 23 stations by `scripts/crosscheck_grid.py`; longitudes between the last and first node wrap through the meridian rather than being clamped). No elevation or lapse-rate adjustment is applied; station elevation and the model grid-cell elevation are listed in the station metadata so that readers can judge representativeness. A lapse-rate-corrected variant is planned as a further sensitivity column, never as the headline.

## 3. Truth

The truth for each station-day is the daily maximum and minimum temperature in the **NWS Daily Climate Report (AFOS product `CLIxxx`)**, taken from the `YESTERDAY` block of the **first CLI issued after local midnight** — the final report for the day. Intermediate same-day reports (`TODAY ... VALID AS OF ...`) are ignored. Later corrected reports are stored with their issuance time in a `revised` column but **do not change published scores** ("first-final" policy). The NWS Preliminary Monthly Climate Data (CF6) is used for month-end reconciliation and to fill gaps; hourly station observations from `api.weather.gov` are used only as a flagged fallback and for quality control (hourly sampling misses peaks by ~1 °F).

Only the **final** CLI value is used. A same-day preliminary report is discarded outright — it is never used even as a last-resort fallback, and the priority order is: final CLI → CF6 → hourly observations (always flagged `obs_fallback`).

**Units and rounding.** NWS reports whole degrees Fahrenheit; forecasts are stored in °C at float precision. Errors are computed in °C and displayed in °F. The quantisation is a genuine rounding, not a truncation: ASOS holds the daily extremes internally in tenths of a degree Celsius and reports temperature "rounded to the nearest degree Fahrenheit", with "all mid-point temperature values … rounded **up** (e.g. +3.5 °F rounds up to +4.0 °F; −3.5 °F rounds up to −3.0 °F; while −3.6 °F rounds to −4.0 °F)" ([ASOS User's Guide](https://www.weather.gov/media/asos/aum-toc.pdf), §3.1.3). The published truth therefore carries a half-degree-Fahrenheit uncertainty of its own — the true extreme lies in `[F − 0.5 °F, F + 0.5 °F)` — which is not propagated into the confidence intervals and is one of the limitations in §10. The ±1/±2/±3 °F hit-rate thresholds are **inclusive**: a day with `|error| = 1.000 °F` counts as a ±1 °F hit.

## 4. Scores

For each `station × model × initialization × lead_day × variable` and for the windows *last 30 days, last 90 days, last 365 days, all available*, we publish:

| Score | Column | Definition |
|---|---|---|
| n | `n` | number of scored days |
| Stations per day | `n_stations` | mean number of stations behind the daily value (1 for a station row; see below for `ALL`) |
| Flagged days | `n_flagged` | scored days whose truth carries a `qc_flag` (§6); they are **included** in every score |
| MAE | `mae` | mean absolute error, forecast − observed |
| Bias (ME) | `bias` | mean error, forecast − observed (positive = too warm) |
| RMSE | `rmse` | root mean square error |
| Hit rate ±1 / ±2 / ±3 °F | `hit1f`, `hit2f`, `hit3f` | fraction of days with absolute error **≤** the threshold |
| Debiased MAE | `mae_debiased` | `mean(|e − mean(e)|)` over the window: the random part of the error, with the model's own systematic offset removed |
| Skill vs persistence | `skill_persistence` | `1 − MAE_model / MAE_persistence`, computed on the days both have |
| Debiased skill | `skill_persistence_debiased` | the same ratio computed from `mae_debiased`, both series de-meaned over the common days |

Windows with n < 30 are shown greyed and excluded from rankings. Scores are computed only within each model's own data-availability period; every page shows the availability bar per model.

**Persistence baseline (lagged persistence).** For target day `D` at lead day `L` the baseline is the *observed* value of day `D − L` — the freshest observation a forecaster issuing at that lead already had. The alternative "yesterday's observation at every lead" was rejected: at lead 7 it would hand the baseline six days of information the model never saw, making it artificially hard to beat exactly where the skill curve is most interesting. Lead day 0 has no baseline (`D − 0` is the target day itself).

Because the baseline is an observed extreme while the model column is a 6-hourly *sampled* extreme (§2.3), and because a 0.25° grid cell is not the station, `skill_persistence` mixes forecast error with a constant systematic offset that differs by station (GFS, 00 UTC, lead day 1, Tmax, 30-day window, n = 28 at each station: `bias` = −1.53 °C at KDEN, −1.17 °C at KORD, +1.68 °C at KNYC — and `skill_persistence` is +0.31, +0.14 and −0.35 respectively, while `skill_persistence_debiased` is +0.50, +0.36 and +0.18). `skill_persistence_debiased` removes each series' own mean error over the common days and is the number to use when asking whether a model tracks day-to-day variability better than persistence; `skill_persistence` remains the honest end-to-end number for "how far off was the raw field".

**`ALL` rows.** The daily value of an `ALL` row is first averaged across the stations that have a value on that day, and the statistics and bootstrap then run over days exactly as for a single station. Pooling all station-days instead would weight a day with 23 stations 23× a day with one and would destroy the day as an exchangeable resampling unit, so the day-mean is the published aggregate and the pooled variant is not published. The cost is that two models with different station coverage produce `ALL` rows built from different station mixtures; `n_stations` is published so that this is visible, and cross-model claims should be read off the pairwise table.

## 5. Uncertainty

Every score carries a **95 % confidence interval from a moving-block bootstrap over days** (1000 resamples, block length 7 days). `mae`, `bias`, `rmse` and `hit1f` carry intervals (`*_ci_low` / `*_ci_high`).

- **Why blocks.** Consecutive daily forecast errors are not independent — a persistent synoptic regime biases many days the same way. An i.i.d. day resample treats them as independent and produces intervals that are too narrow. A circular moving-block resample draws runs of 7 consecutive days, so within-run dependence survives the resampling. On the current archive the block intervals are a median 1.15× wider than i.i.d. ones over groups with n ≥ 20, and 1.48× wider over the long (n ≥ 100) baseline series where the effect is measurable. The block length is clipped so that a short window always has at least four blocks to draw from.
- **How it is computed, and the approximation involved.** One resample-count matrix is drawn per window and shared by every group; each group's statistic is a self-normalised weighted mean over the days it actually has. Conditional on its realised size this is a resample of that group's own days, with a randomised resample size whose first-order effect the ratio form removes. `scripts/crosscheck_bootstrap.py` measures the residual against a textbook per-group percentile bootstrap on the real archive: the interval widths agree to 0.99–1.06× (median 1.015), i.e. within Monte-Carlo noise. Sharing the matrix is what makes the single-model and paired intervals mutually consistent.
- **Model-vs-model comparisons use a paired bootstrap on the day-by-day difference series** over the days on which both models have a valid forecast; two models are reported as distinguishable only when the CI of the MAE difference excludes zero. Overlapping single-model intervals are not evidence of equivalence.
- **Not included in any interval:** the ±0.5 °F quantisation of the truth (§3), representativeness error between the station and the grid cell, and the multiplicity of comparing many models at many leads. Intervals are marginal, not simultaneous.

## 6. Missing data and quality control

- If any of the four common samples is unavailable for a model run, that station-day is missing for that model and is recorded as an explicit row with a `missing_reason`; nothing is silently skipped.
- Per-model headline scores use all of that model's valid days (n is shown). Pairwise comparisons use the intersection.
- If the CLI value and the hourly-observation-derived extreme differ by more than 2 °F the day is flagged (`qc_flag`) and reported with the flag; flagged days remain in the scores, and the count of flagged days is published per score row as `n_flagged`. A day whose truth came from the hourly-observation fallback rather than CLI is flagged `obs_fallback` and also counted there. For an `ALL` row a day counts as flagged as soon as any of its stations is.
- Missing (`M`) or trace values in CLI make the day missing for that variable.

## 7. Fairness statement

These are **raw model outputs on the native 0.25° grid, without MOS, bias correction, downscaling or any post-processing**. They are not equivalent to the products end users receive from a weather service or app, and the scores here **understate operational forecast quality**. Model runs initialized from different analyses (AIWP models from GFS vs IFS initial conditions) are listed as separate models.

**Version segmentation.** A `model_version` identifier (`ifs-gpid161`, `gfs-0p25`, `FOUR_v200`, …) is stored with every extracted value. When a model changes cycle or weights the change is a **segment boundary and scores are never aggregated across it**: for each `model_id` the scoring window is truncated to the most recent contiguous `model_version` segment, and every score row publishes that `model_version` together with `segment_start`, the first initialization inside the segment. Runs whose version could not be determined (`"unknown"`) never open or close a segment. Known boundaries that this rule will bite as the backfill deepens: FourCastNet v100 → v200 on 2023-10-31, and the IFS `gpid` 158 → 161 change. Older segments remain in the archived `daily_forecasts` table and can be re-scored, but they do not appear in the published aggregates.

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
2. No elevation adjustment; mountain and coastal stations may show representativeness error that is not model error. Because the persistence baseline is an observed extreme and the model column is a sampled one, this offset lands entirely in `skill_persistence`; use `skill_persistence_debiased` alongside it (§4).
3. Truth is a single station observation with its own uncertainty (±0.5 °F rounding, occasional sensor issues), and that uncertainty is **not** propagated into the confidence intervals (§5).
4. Only two initializations and one variable.
5. Twenty-three stations in one country.
6. Confidence intervals are marginal. Nothing corrects for the number of models × leads × stations being compared, so some "distinguishable" pairs at the 95 % level are chance.
7. `ALL` rows average whichever stations a model has that day, so two models' `ALL` rows can rest on different station mixtures (§4); `n_stations` exposes this.
8. Native extremes (§2.4) overhang the climatological day by up to 6 h at 15 of the 23 stations. They are a diagnostic and never enter a ranking.

## Changelog

- **0.2 (2026-08-30)** — methodology and statistics review.
  - §2.4 native extremes: the "buckets entirely inside the day" rule (which yielded values only at −6 h stations) is replaced by "the contiguous overlapping run, at most one crossing bucket at each end, overhang ≤ 6 h"; the overhang is documented per station offset. Native extremes remain diagnostic.
  - §3: preliminary CLI reports are dropped rather than ranked below the observation fallback. NWS rounding (nearest °F, mid-points up; ASOS internal tenths of °C) documented; hit-rate thresholds stated as inclusive.
  - §4: new published columns `n_stations`, `n_flagged`, `mae_debiased`, `skill_persistence_debiased`, `rmse_ci_low/high`, `hit1f_ci_low/high`, `model_version`, `segment_start`. The lagged-persistence baseline and the day-mean `ALL` aggregate are stated as rulings with their rejected alternatives.
  - §5: the day bootstrap is now a **circular moving-block** bootstrap (block 7 days). The shared-resample-matrix approximation is documented and measured against a textbook per-group bootstrap.
  - §7: version segmentation is implemented, not merely asserted — the scoring window is truncated to the latest `model_version` segment.
- **0.1 (2026-08-30)** — initial pre-release methodology.
