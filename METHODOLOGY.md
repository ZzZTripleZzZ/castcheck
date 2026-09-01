# CastCheck Methodology

**Version 0.3.1 — 2026-08-31 (pre-release).** This document is versioned together with the data. Any change that alters a published score bumps the version and is listed in the changelog at the end. v0.3 is the response to an external methodological review (`docs/06-external-review-v02.md`); it changes what the headline number *is*, so v0.2 scores are not comparable to v0.3 scores.

CastCheck is an independent, automated, station-level verification of publicly available weather forecasts. It answers one narrow question, every day, with full history: *at this station, at this valid time, how far was each model's raw 2 m temperature forecast from what the National Weather Service instrument actually measured?*

## 1. Scope (v0.3)

| Dimension | v0.3 |
|---|---|
| Variable | 2 m air temperature: the instantaneous value at 00/06/12/18 UTC (**headline**), and daily maximum/minimum derived from it |
| Stations | 23 U.S. airport / first-order stations with an NWS daily climate report (CLI) and an ASOS hourly record; see `config/stations.yaml` |
| Models | ECMWF IFS HRES, ECMWF AIFS Single, NCEP GFS, and the NOAA/CIRA AIWP operational runs of GraphCast, Pangu-Weather, FourCastNet v2 and Aurora (each from both GFS and IFS initial conditions) |
| Initializations | 00 UTC and 12 UTC, scored separately |
| Lead | Lead day 0–9 (all computed); the headline pages show lead days 1, 3, 5 and 7 |
| Truth | ASOS routine METAR at the synoptic hour (instantaneous); NWS Daily Climate Report (CLI), first final issuance (daily extremes) |
| Baseline | Lagged persistence: the observation of the **same functional** `lead` days earlier |

Everything else — precipitation, wind, ensembles/probabilities, non-U.S. stations, post-processed products — is out of scope for now and will be added only after the temperature line has run for 90 days and survived public review.

### 1.1 How the 23 stations were chosen

The station list is **not a random or a climatologically stratified sample**, and the reader must not treat the `ALL` row as an estimate of national forecast skill. The selection rule, stated in full:

1. **Overlap with the publicly traded city-temperature markets.** Every station is the settlement station of a listed daily-temperature contract (the `market_city` column of `config/stations.yaml`, called `kalshi` before v0.3). This is why the list exists at all: those are the places where an independent, timestamped, non-revisable forecast record has an audience that will check it.
2. **Large first-order airport stations.** All 23 are ASOS/AWOS-equipped major airports (or, for New York, the Central Park first-order site) with a continuous NWS CLI product and an hourly archive going back years. Small, intermittently reporting or recently relocated stations were excluded.
3. **Coverage of the four contiguous-U.S. time zones and of coastal / inland / high-elevation regimes**, so that the LST-day arithmetic (§2.1), the lead-hour asymmetry (§2.5) and the representativeness error (§2.6) are all exercised rather than hidden.

The consequences for external validity are real and are not hidden: the set over-weights the northeastern corridor and the dry Southwest, contains no rural, mountain-valley, Great Plains or non-CONUS site, and every station is an airport — a class of site with its own microclimate (large paved surfaces, exposed sensors, no canopy). **Extrapolating these numbers to "U.S. forecast skill", to non-airport locations, or outside North America is not supported.** Per-station numbers are published precisely so that a reader can see how much the aggregate depends on the mix.

## 2. Definitions

### 2.1 Climatological day
A **climatological day** is midnight-to-midnight in **local standard time (LST)**, exactly as the NWS CLI product defines it. Daylight-saving time is *not* applied: during DST the climatological day runs from 01:00 to 01:00 local clock time. Each station carries a fixed standard UTC offset (e.g. New York −5 h all year). A day is the half-open UTC interval `[day_start, day_end)`.

This matches the instrument: the ASOS Daily Summary Message, from which the CLI daily maximum and minimum are taken, "runs from 00:00 to 23:59 LST" ([ASOS User's Guide](https://www.weather.gov/media/asos/aum-toc.pdf), §3.1.3). Consequently the DST changeover dates are ordinary 24-hour days here, and Phoenix (−7 h, never on DST) is handled by exactly the same arithmetic as Denver.

### 2.2 Common samples
For every model run and station we take the **instantaneous 2 m temperature at forecast valid times of 00, 06, 12 and 18 UTC**. All models in scope provide these steps. Models with finer output (IFS at 3-hourly steps out to 144 h) are deliberately *not* sampled more finely for the headline scores, so that every model is evaluated on an identical sampling.

For any U.S. station, each climatological day contains exactly four common samples (e.g. for a −5 h station the day `[05Z, 05Z+24h)` contains 06Z, 12Z, 18Z and 00Z of the following UTC date).

### 2.3 What is scored (v0.3)

v0.2 made the sampled daily extremes the headline and verified them against the *true* NWS extremes. External review showed that this is not a forecast-error metric: at a −5 h station the true daily minimum occurs around 10–11 UTC and is never sampled, so the Tmin "error" was almost entirely a definition artefact (28 days of same-signed error, MAE ≡ bias, 0 % hit rate). Worse, the size of that artefact **depends on each model's own diurnal amplitude** — a model with a flatter diurnal cycle is penalised more — and model diurnal amplitude is exactly what this site is trying to report. The metric and the measurand were confounded. v0.3 separates them into four named functionals.

**Headline — instantaneous 2 m temperature (`t2`).**
```
err(model, station, valid_time) = T2_forecast(valid_time) − T2_observed(valid_time)
```
at the four common instants 00/06/12/18 UTC, against the ASOS routine METAR closest to that instant (§3.1). There is no extreme, no window and no daily aggregation inside the number: it is a forecast error and nothing else. A day's `t2` score is the **mean of the four instants' functionals** (|e|, e, e², hit indicators averaged separately), so the resampling unit stays the day. `t2_00z`, `t2_06z`, `t2_12z`, `t2_18z` publish the same thing one instant at a time; their diurnal structure is the diagnostic §10 asks for.

**Like-for-like sampled extremes (`tmax_s`, `tmin_s`).**
```
Tmax_s_forecast(day) = max of the four forecast samples in the day
Tmax_s_observed(day) = max of the four *observed* samples at the same four instants
```
Both sides use the same four instants, so the sampling penalty cancels identically and what remains is forecast error in the daily-extreme functional. A day is scored only if all four forecasts *and* all four observations are present (`n_samples = 4` and `n_obs_samples = 4`).

**Business-relevant extremes (`tmax_cli`, `tmin_cli`) — secondary.** The same four forecast samples against the NWS CLI daily maximum/minimum: what a user of a daily-max product actually experiences. **This column contains a sampling penalty as well as forecast error, and the penalty is not the same for every model** — it is roughly the amount by which each model's own diurnal cycle overshoots its four sampled points, so a model with a larger diurnal amplitude gets a smaller penalty. The claim made in v0.2 §2.3, that the bias "applies identically to every model and is therefore fair for comparing models", was wrong and is withdrawn. These columns are labelled as secondary everywhere they appear and must not be used to rank models.

**Native extremes vs CLI (`tmax_native_cli`, `tmin_native_cli`) — secondary, and only for the models that have the field.** See §2.4. This is the one comparison in which the sampling penalty is absent on the forecast side too, which makes it the sharpest available test of how much of the `*_cli` gap is sampling (§10.2).

### 2.4 Native extremes (diagnostic only)
Some models output time-window extremes directly: IFS `mx2t3/mn2t3` (≤144 h) and `mx2t6/mn2t6` (>144 h), GFS `TMAX/TMIN` over 6-hour buckets. These buckets are anchored to whole multiples of the bucket length from 00 UTC, so they align with a climatological day only when the station's standard offset is a multiple of the bucket length — in practice only the −6 h (CST) stations. Requiring the buckets to lie *entirely* inside the day (the v0.1 rule) therefore produced a value at 8 of 23 stations and `NULL` at the other 15, which is not a usable diagnostic.

**v0.2 rule.** The day is covered by the **contiguous run of buckets that overlaps it**: every bucket inside the day, plus at most one crossing bucket at each end. The run must be gap-free and non-overlapping, must cover the whole day, and may overhang it by at most 6 h in total. The overhang is a deterministic property of the station and the bucket length, not of the date:

| station standard offset | 3 h buckets (IFS ≤144 h) | 6 h buckets (IFS >144 h, GFS) |
|---|---|---|
| −6 h (CST) | 0 h — exact tiling | 0 h — exact tiling |
| −5 h, −7 h, −8 h | 3 h (2 h before + 1 h after, or 1 h + 2 h) | 6 h |

`castcheck.derive.native_overhang_hours(std_offset_h, bucket_h)` returns it in closed form, and since v0.3 every `daily_forecasts` row also publishes the overhang it actually realised as `native_overhang_h`, so a reader can filter on it. A native maximum that is contaminated by an overhang can only be *too extreme*, never too flat, and the contamination is the same for every model at a given station, so it does not distort a model-vs-model comparison at that station. It does make the native column slightly less comparable *across* stations, which is one more reason it is a diagnostic. The day that straddles the IFS 144 h change of bucket length is accepted if the 3 h and 6 h buckets happen to form one gap-free run, and is `NULL` otherwise.

**Native extremes are never mixed into the headline ranking**, because the AI models have no equivalent field and would be penalised purely for a missing output type. They are published as the separate secondary variables `tmax_native_cli` / `tmin_native_cli`, on the models that have them, for the attribution analysis of §10.2.

### 2.5 Lead day
`lead_day = target climatological date − UTC date of the model initialization`. A 00 UTC run on 30 August scored against the 31 August climatological day is lead day 1. Because 00 UTC and 12 UTC runs reach the same target day at different forecast hours, the two initializations are always scored and displayed separately.

A day is scored only when **all four** of its common samples fall inside `(init, init + max_h]`. Two consequences are visible in the tables and are not errors: a 12 UTC run never produces lead day 0 for a U.S. station (its 06 UTC sample precedes the run), and at a 240 h horizon a 00 UTC run reaches lead day 9 only for −5 h and −6 h stations — the −7 h and −8 h stations stop at lead day 8, so the `ALL` row at lead day 9 rests on fewer stations. The number of stations behind every `ALL` row is published as `n_stations`.

**The same `lead_day` is a different forecast range at different longitudes.** `lead_day` counts *climatological days*, and a climatological day starts at a different UTC hour in each time zone, so the four instants of "lead day 0" sit at different forecast hours for an eastern and a western station. For a 00 UTC run:

| station offset | lead day 0 covers | lead day 1 covers |
|---|---|---|
| −5 h (EST) | F06, F12, F18, F24 | F30 … F48 |
| −6 h (CST) | F06 … F24 (day starts 06 UTC) | F30 … F48 |
| −8 h (PST) | F12, F18, F24, F30 | F36 … F54 |

A −8 h station is therefore forecast about **6 h further ahead** than a −5 h station at the same nominal lead day (about 3 h further than the 23-station mean), and the `ALL` row mixes those ranges together. Where the difference matters — a lead-curve slope of roughly 0.02–0.05 °C h⁻¹ at short range implies a tenth of a degree or so across the CONUS spread — read the per-station pages, whose lead days are internally consistent, rather than the `ALL` row. The instantaneous variables publish `lead_h` in the underlying `forecast_values` download, so the mixture can be undone by anyone who wants to.

### 2.6 Station value
The model value at a station is obtained by **bilinear interpolation from the four surrounding grid nodes** of the model's native 0.25° latitude–longitude grid (primary). The **nearest-node** value is computed and published alongside as a sensitivity variant. Sources use two different conventions — ECMWF stores longitude −180…179.75, GFS and AIWP store 0…359.75, and all three store latitude descending — and the extraction normalises both to give bit-identical answers (verified to 1×10⁻¹³ K on a synthetic field at all 23 stations by `scripts/crosscheck_grid.py`; longitudes between the last and first node wrap through the meridian rather than being clamped). No elevation or lapse-rate adjustment is applied. `config/stations.yaml` (and the `stations.csv` download) carry the station elevation `elev_m`, the mean elevation of the 0.25° grid cell containing it `grid_elev_m`, and their difference `dz_m = elev_m − grid_elev_m`, together with the first-order lapse-rate magnitude `|dz_m| × 6.5 K/km`, so that readers can judge representativeness themselves; `grid_elev_m` comes from the public-domain ETOPO 2022 60-arc-second grid and is frozen at build time. A lapse-rate-corrected variant is planned as a further sensitivity column, never as the headline.

## 3. Truth

### 3.1 Instantaneous truth (headline)

The truth for an instantaneous forecast is the **routine ASOS METAR closest to the synoptic hour, within ±35 minutes**, preferring the scheduled :51–:56 observation. It is stored in the `truth_instant` table with the timestamp of the report actually used (`obs_time`), the number of reports found inside the window (`n_reports`) and a `qc_flag` (`""`, `no_report`, `gap_gt35min`, `suspect`). The source is the Iowa Environmental Mesonet ASOS archive (`report_type=3`, i.e. routine hourly METARs only, no SPECIs) with `api.weather.gov` covering the most recent week. ASOS reports temperature in whole °F (and tenths of °C in the coded remark); the value is stored as reported and converted to °C.

Two limitations are inherent and are not corrected: the METAR is an instantaneous 1-minute-average value, whereas a model's `t2` is a grid-cell mean over 0.25°, and the report can be up to 35 minutes away from the nominal instant (at 5 °C h⁻¹ that is up to 3 °C on a fast-warming morning). The offset is a property of the station and the hour, not of the model, so it does not favour any model; it does inflate every model's MAE by a common amount.

### 3.2 Daily-extreme truth

The truth for each station-day is the daily maximum and minimum temperature in the **NWS Daily Climate Report (AFOS product `CLIxxx`)**, taken from the `YESTERDAY` block of the **first CLI issued after local midnight** — the final report for the day. Intermediate same-day reports (`TODAY ... VALID AS OF ...`) are ignored. Later corrected reports are stored — a `revised` flag plus the corrected values in `revised_tmax_f`/`revised_tmin_f` — but **do not change published scores** ("first-final" policy). The NWS Preliminary Monthly Climate Data (CF6) is used for month-end reconciliation and to fill gaps; hourly station observations from `api.weather.gov` are used only as a flagged fallback and for quality control (hourly sampling misses peaks by ~1 °F).

Only the **final** CLI value is used. A same-day preliminary report is discarded outright — it is never used even as a last-resort fallback, and the priority order is: final CLI → CF6 → hourly observations (always flagged `obs_fallback`).

**Units and rounding.** NWS reports whole degrees Fahrenheit; forecasts are stored in °C at float precision. Errors are computed in °C and displayed in °F. The quantisation is a genuine rounding, not a truncation: ASOS holds the daily extremes internally in tenths of a degree Celsius and reports temperature "rounded to the nearest degree Fahrenheit", with "all mid-point temperature values … rounded **up** (e.g. +3.5 °F rounds up to +4.0 °F; −3.5 °F rounds up to −3.0 °F; while −3.6 °F rounds to −4.0 °F)" ([ASOS User's Guide](https://www.weather.gov/media/asos/aum-toc.pdf), §3.1.3). The published truth therefore carries a half-degree-Fahrenheit uncertainty of its own — the true extreme lies in `[F − 0.5 °F, F + 0.5 °F)` — which is not propagated into the confidence intervals and is one of the limitations in §10. The ±1/±2/±3 °F hit-rate thresholds are **inclusive**: a day with `|error| = 1.000 °F` counts as a ±1 °F hit.

### 3.3 Plausibility check on the daily extremes (v0.3.1)

First-final (§3.2) is what makes the daily truth reproducible, and it is kept. It has one failure mode that only became visible once `truth_instant` existed: a *garbled* first report is scored as truth. KLAX 2025-02-16 was issued with `MINIMUM 11R` and corrected to 49 six hours later, so the published minimum sat 38 °F below every observation taken that night; KLAX 2024-05-05 (17 → 53), KEWR 2026-03-04 (24 → 37) and KDCA 2025-05-18 (maximum 87 → 78) are the same thing. That is not a first-final ruling, it is a decode error surviving into the scores.

Every CLI extreme is therefore checked against the four sampled observations of its own climatological day, and only on days where all four are present and none is flagged `suspect`. The four samples are part of the same trace as the daily extreme, so two statements hold:

- **Impossible.** A daily maximum below the sampled maximum (or a minimum above the sampled minimum) by more than **1.5 °F** cannot happen. The tolerance covers whole-°F rounding on both sides and the up-to-35-minute offset between a synoptic instant and the METAR that represents it.
- **Excursion.** The extreme lying *beyond* the sampled extreme is ordinary — a dawn minimum falls between the 06 and 12 UTC samples — so distance alone proves nothing. Measured on this archive the genuine excursions reach 20 °F on the maximum (dry high-plains afternoons at KDEN) and 25 °F on the minimum (winter frontal passages at KOKC/KORD), with no gap separating them from the decode errors. A pure magnitude threshold cannot tell the two apart, so an excursion beyond **10 °F** is acted on *only when there is corroborating evidence*: a corrected issuance, or a CF6 value, that passes the same check.

Resolution, per variable, in fixed order: the corrected value if it passes (`cli_implausible_revised_used`), else the CF6 value if it passes (`cli_implausible_cf6_used`), else the value is dropped to `NaN` so the day leaves the scores (`cli_implausible_dropped`).

An **uncorroborated excursion is kept**, not dropped, as long as it stays inside the envelope of excursions that are known to be real: dropping on distance alone would remove precisely the most extreme days, which is the worst possible bias for a verification site. That envelope is measured, not assumed — the widest genuine excursion anywhere in the 2024-2026 archive is **25 °F** (KOKC 2024-02-27: samples 62/64/73/74 °F, reported minimum 37 °F — 25 °F below the lowest sample, a February frontal passage after the last sample of the day; no correction was ever issued, because nothing was wrong), against 20 °F for the maximum (KDEN 2025-07-17, KEWR 2025-12-07). Beyond that bound the value is dropped even with nothing to replace it: an isolated reading further outside the envelope than three years of observations support is worth less as truth than the station-day is worth as coverage. KATL 2026-04-14 is the case this covers — a minimum of 32 °F reported against samples of 65/63/82/80 °F, 31 °F below the lowest of them, with a correction that touched only the maximum, so neither the corrected issuance nor CF6 could supply a replacement.

Over 2024-01-01 → 2026-08-31 this touches **14 station-days out of 22,328** (0.06 %): 4 repaired from the correction — exactly the four cases above — 9 dropped as impossible, and 1 dropped as an uncorroborated excursion past the observed envelope. The check is idempotent (a repaired value passes it) and is re-run over a trailing 15-day window on every daily pass, since corrections arrive late. Rows whose published value it changed carry `methodology_version = "0.3.1"`; the superseded value remains in the NWS product archive and in the shard's git history.


## 4. Scores

For each `station × model × initialization × lead_day × variable` and for the windows *last 30 days, last 90 days, last 365 days, all available*, we publish:

| Score | Column | Definition |
|---|---|---|
| n | `n` | number of scored days |
| Stations per day | `n_stations` | mean number of stations behind the daily value (1 for a station row; see below for `ALL`) |
| Flagged days | `n_flagged` | scored days whose truth carries a `qc_flag` (§6); they are **included** in every score |
| MAE | `mae` | mean absolute error |
| Bias (ME) | `bias` | mean error, forecast − observed (positive = too warm) |
| RMSE | `rmse` | root mean square error |
| Hit rate ±1 / ±2 / ±3 °F | `hit1f`, `hit2f`, `hit3f` | fraction of days with absolute error **≤** the threshold |
| Out-of-sample debiased MAE | `mae_debiased`, `n_debiased` | `mean(|e_t − b̂_t|)` where `b̂_t` is the mean error over the **previous** 30 scored days |
| Days behind the skill ratio | `n_common` | days on which both the model and the baseline have a value |
| Baseline MAE on those days | `mae_persistence_common` | the denominator of `skill_persistence`, on exactly those `n_common` days |
| Skill vs persistence | `skill_persistence` | `1 − mae_over_common / mae_persistence_common` |
| Debiased skill | `skill_persistence_debiased` | the same ratio computed from the out-of-sample debiased errors of both series |

Windows with n < 30 are shown greyed and excluded from rankings. Scores are computed only within each model's own data-availability period; every page shows the availability bar per model.

**The skill column is recomputable from its own row.** In v0.2 the skill used the *intersection* days as denominator while the table's persistence row showed its own, longer `n` — so a reader who divided the two printed numbers got a different answer from the printed skill. That is fixed by publishing `n_common` and `mae_persistence_common`: `skill_persistence` is exactly `1 − (model MAE over the n_common days) / mae_persistence_common`, and where `n = n_common` the model's own `mae` is that numerator. The site prints `mae_persistence_common (n_common)` beside every skill value.

**Persistence baseline (lagged persistence, of the same functional).** For target day `D` at lead day `L` the baseline is the observation of day `D − L` — the freshest observation a forecaster issuing at that lead already had. The alternative, "yesterday's observation at every lead", was rejected: at lead 7 it would hand the baseline six days of information the model never saw, making it artificially hard to beat exactly where the skill curve is most interesting. Lead day 0 has no baseline (`D − 0` is the target day itself).

The baseline is always **the same functional as the numerator**: for `t2*` it is the observation at the same UTC hour `L` days earlier; for `tmax_s`/`tmin_s` it is the observed *sampled* extreme of day `D − L`; for the `*_cli` variables it is the CLI extreme of day `D − L`. v0.2 used the CLI daily extreme for every variable, so the numerator carried the extreme-sampling penalty and the denominator did not — which made `skill_persistence` negative for essentially every model at lead 1 and turned the site's most prominent column into a statement about the sampling rule rather than about forecast quality. With the like-for-like baseline the ratio is again a forecast statement.

What remains inside `skill_persistence` is representativeness: a 0.25° grid cell is not the station, so a station with a large `dz_m` (§2.6) still shows a systematic offset in the numerator only. `skill_persistence_debiased` removes each series' own **out-of-sample** mean error and is the number to use when asking whether a model tracks day-to-day variability better than persistence; `skill_persistence` remains the end-to-end number for "how far off was the raw field".

**Debiasing is out of sample.** `mae_debiased` estimates each series' bias on the trailing 30 *scored* days strictly **before** the day being scored, requires at least 15 of them, and applies it forward; days without enough history are excluded and counted in `n_debiased`. The v0.2 column fitted one constant on the same days it then scored, which is a one-parameter in-sample fit and flatters exactly the models with the largest constant offset (it moved one model's skill from −0.92 to +0.48 on its own). Out of sample the estimate carries its own variance, so `mae_debiased` is no longer guaranteed to be below `mae` — that is the honest version of the number.

**`ALL` rows.** The daily value of an `ALL` row is the cross-station mean of **each functional separately** — |e|, e, e², each of the three hit indicators and the debiased |e| are each averaged over the stations that have a value that day — and the statistics and bootstrap then run over days exactly as for a single station. It is emphatically *not* the absolute value of a mean error, or a mean of daily errors that is later put through |·|: averaging first and taking the absolute value second would let a +2 °C station cancel a −2 °C station and report an `ALL` MAE of zero. Pooling all station-days instead would weight a day with 23 stations 23× a day with one and would destroy the day as an exchangeable resampling unit, so the day-mean is the published aggregate and the pooled variant is not published. The cost is that two models with different station coverage produce `ALL` rows built from different station mixtures; `n_stations` is published so that this is visible, and cross-model claims should be read off the pairwise table.

## 5. Uncertainty

`mae`, `bias`, `rmse` and `skill_persistence` carry **95 % percentile intervals from a circular moving-block bootstrap over days** (1000 resamples, block length 7 days); the hit rates carry **Wilson score intervals**.

- **The bootstrap is drawn on each group's own realised date axis.** v0.2 drew one resample-count matrix per *window* and shared it across every group, evaluating each group as a self-normalised weighted mean over the days it happened to have. For dense groups that is a good approximation; for sparse ones it is not, and it produced the visible contradiction that four windows over *the same* 28 realised days gave four different intervals for one point estimate. v0.3 draws the resample directly on the days the group actually has, so an interval depends on the data and on nothing else. Implementation detail with a statistical consequence: groups are bucketed by their realised date *set* and the resample matrix is seeded from a stable hash of that set, so two groups with the same scored days — including the same group seen through two different windows — get the same resample and therefore, on the same data, bit-identical intervals. `tests/test_verify.py::test_identical_day_samples_give_identical_intervals_in_every_window` is the regression test.
- **No interval below 28 days or 4 blocks.** A percentile interval from fewer than four independent blocks is not an interval, it is noise with brackets. Groups with `n < 28`, or with fewer than 4 blocks of the chosen length, publish `NaN` and the site shows "—". The point estimate is still published with its `n`.
- **Why blocks.** Consecutive daily forecast errors are not independent — a persistent synoptic regime biases many days the same way. An i.i.d. day resample treats them as independent and produces intervals that are too narrow. A circular moving-block resample draws runs of 7 consecutive days, so within-run dependence survives the resampling. On the current archive the block intervals are a median 1.15× wider than i.i.d. ones over groups with n ≥ 20, and 1.48× wider over the long (n ≥ 100) baseline series where the effect is measurable. `scripts/crosscheck_bootstrap.py` recomputes a textbook per-group percentile bootstrap on the real archive and compares it to the published intervals.
- **Hit rates use Wilson score intervals, not the bootstrap.** A proportion near 0 or 1 has a degenerate bootstrap: 28 days with no hit resample to no hit every time, and v0.2 printed `0 % [0 %, 0 %]`, which reads as certainty where there is none. The Wilson score interval for `k` successes in `n` days is well behaved at the boundaries — 0 of 28 becomes `[0 %, 12 %]`. For the pooled `t2` variable and for `ALL` rows the daily "hit" is itself a mean (over four instants, or over stations), so `k = n × hit_rate` is fractional and the interval treats the *day* as the independent Bernoulli unit; it is exact for a single-station, single-instant variable and an approximation elsewhere. It is used for `hit1f`; `hit2f`/`hit3f` are published without intervals.
- **Model-vs-model comparisons use a paired bootstrap on the day-by-day difference series** over the days on which both models have a valid forecast, drawn on that pair's own common date set. Each pair publishes `mae_diff`, its interval, and a two-sided bootstrap p-value `p_boot`. Overlapping single-model intervals are not evidence of equivalence.
- **Multiplicity is corrected where it is displayed.** The full pairwise table is hundreds of thousands of rows; at a 95 % single-comparison level, thousands of them are "significant" by chance. The uncorrected verdict is still published, but under the name `distinguishable_uncorrected`, and the site never marks a difference on it. What the site marks is `distinguishable_holm`: a **Holm–Bonferroni** correction applied to the family that a reader actually sees at once — every comparison against the leading model within one displayed table (one station, init, lead day, variable, method and window). Comparisons outside that family are `False` by construction. Holm controls the family-wise error rate within a table; it does *not* correct across tables, and a reader who scans 20 station pages looking for the one that supports a claim is doing multiplicity we cannot correct for them.
- **Not included in any interval:** the ±0.5 °F quantisation of the truth (§3.2) and the ±35 min timing tolerance of the instantaneous truth (§3.1), representativeness error between the station and the grid cell (§2.6), and multiplicity across displayed tables. Intervals are marginal within a table, not simultaneous across the site.

## 6. Missing data and quality control

- If any of the four common samples is unavailable for a model run, that station-day is missing for that model and is recorded as an explicit row with a `missing_reason`; nothing is silently skipped.
- Per-model headline scores use all of that model's valid days (n is shown). Pairwise comparisons use the intersection.
- If the CLI value and the hourly-observation-derived extreme differ by more than 2 °F the day is flagged (`qc_flag`) and reported with the flag; flagged days remain in the scores, and the count of flagged days is published per score row as `n_flagged`. A day whose truth came from the hourly-observation fallback rather than CLI is flagged `obs_fallback` and also counted there. For an `ALL` row a day counts as flagged as soon as any of its stations is.
- Missing (`M`) or trace values in CLI make the day missing for that variable.
- A CLI extreme that contradicts the day's own four sampled observations is repaired from the corrected issuance or CF6, or — when it is physically impossible and nothing can replace it — dropped for that variable (§3.3). All four outcomes are recorded in `qc_flag` (`cli_implausible` plus `…_revised_used` / `…_cf6_used` / `…_dropped`) and counted in `n_flagged` like every other flag. 14 station-days in the 2024-2026 archive are affected.

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

## 10. Known limitations (v0.3)

### 10.1 Limitations

1. The `*_cli` variables are computed from four samples per day and therefore understate diurnal extremes; the penalty differs by model (§2.3). They are secondary and are not a ranking.
2. No elevation adjustment; mountain and coastal stations may show representativeness error that is not model error. It lands in the numerator of `skill_persistence` only; use `skill_persistence_debiased` alongside it (§4), and `dz_m` in the station metadata (§2.6) to see which stations are affected.
3. Truth has its own uncertainty — ±0.5 °F rounding on the CLI extremes, up to ±35 min of timing tolerance and a point-vs-grid-cell mismatch on the instantaneous values — and none of it is propagated into the confidence intervals (§5).
4. Only two initializations and one variable.
5. Twenty-three airport stations in one country, chosen for market overlap, not sampled (§1.1). The `ALL` row is an average over that specific set and nothing more.
6. `lead_day` is not a fixed forecast range across longitudes; the `ALL` row mixes ranges that differ by up to 6 h (§2.5).
7. Confidence intervals are marginal within a displayed table. Holm corrects the leader comparisons inside one table (§5); nothing corrects across the site's many tables.
8. `ALL` rows average whichever stations a model has that day, so two models' `ALL` rows can rest on different station mixtures (§4); `n_stations` exposes this.
9. Native extremes (§2.4) overhang the climatological day by up to 6 h at 15 of the 23 stations; `native_overhang_h` publishes the realised value per row.
10. The record is still short. Until each model has ≥ 90 days of parallel record, no ranking on this site should be quoted as a property of the models.

### 10.2 An unattributed empirical observation: the AI models' 18/00 UTC cold bias

Under the four-instant sampling, the AI models show a **larger cold bias at 18 and 00 UTC** — early-to-late afternoon local time at U.S. stations — than at 06 and 12 UTC, and a larger one than the physical models show at the same instants. This is reported here as **an observation about our own measurements, not as a finding about the models**: it is **not yet attributed** to any cause, because we have not separated the candidates. The site states it in exactly those terms and draws no causal conclusion.

Three candidate explanations, none of them excluded by the present data:

1. **Model diurnal amplitude.** The AI models may genuinely produce a flatter diurnal cycle — damped afternoon warming — whether because of their training objective, their resolution, or their land-surface treatment. If so the bias is a real model property.
2. **The extreme-sampling penalty (§2.3).** In the `*_cli` variables a flatter forecast curve is penalised more heavily by the four-instant sampling, so part of any "AI is colder" gap seen in those columns is a property of our metric, not of the model. The `t2_18z` variable is free of that penalty, which is why the diurnal diagnostic is published on the instantaneous variables.
3. **The initial conditions.** The same architecture run from GFS and from IFS analyses differs by 1–2 °C in these tables, which is a large fraction of the effect and points at near-surface temperature in the analysis rather than at the forecast model.

The evidence available today argues **against** a simple "trained on ERA5, which flattens the diurnal range" story on its own: AIFS, trained on ERA5, sits close to the physical IFS HRES, while Aurora and GraphCast — also ERA5-trained — sit several degrees away. If the training label were the mechanism, the four AI models should cluster; they do not.

The analyses that would attribute it, in the order we intend to run them:

- **(a) Per-instant bias curves** for every model at every lead (`t2_00z … t2_18z`) — published as of v0.3, and the first thing to look at.
- **(b) Native vs sampled extremes.** Compare `tmax_native_cli` with `tmax_cli` for the models that have a native field. If the gap between models shrinks sharply under native extremes, the sampling penalty is the dominant term.
- **(c) ERA5 through the same pipeline.** Run ERA5 itself at the same 23 stations through this exact code and score its sampled and CLI extremes. That measures the "training label" ceiling directly.
- **(d) Seasonal and station-class splits** (coastal / inland / high-elevation, summer / winter), since diurnal amplitude errors are strongly seasonal and terrain-dependent.

Until (b), (c) and (d) exist, the correct sentence is: *under four-instant sampling the AI models' afternoon cold bias is larger, and we have not yet separated model diurnal amplitude, sampling penalty and initial conditions.*

## Changelog

- **0.3.1 (2026-08-31)** — CLI plausibility check against the instantaneous observations.
  - §3.3 (new) / §6: the published daily extreme is checked against the four sampled observations of the same day and repaired from the corrected issuance or CF6, or dropped when it is physically impossible and nothing can replace it, or when it lies past the widest excursion three years of observations support (25 °F). First-final is unchanged; only values that contradict the station's own measurements are acted on. 14 station-days affected in the 2024-2026 archive.
- **0.3 (2026-08-31)** — external methodological review (`docs/06-external-review-v02.md`). **Published scores are not comparable to v0.2.**
  - §2.3: the headline is now the **instantaneous** 2 m temperature at the four common instants (`t2`, plus `t2_00z…t2_18z`) against the observation at the same instant. Sampled daily extremes are scored **like for like** against the observed sampled extremes (`tmax_s`, `tmin_s`). The old comparison — sampled forecast extreme vs true CLI extreme — survives as the secondary `tmax_cli`/`tmin_cli`, explicitly labelled as carrying a sampling penalty whose size varies with each model's diurnal amplitude. The v0.2 claim that this penalty "applies identically to every model and is therefore fair for comparing models" is withdrawn.
  - §3.1: new instantaneous truth table `truth_instant` (ASOS routine METAR within ±35 min of the synoptic hour).
  - §2.4: the realised native-extreme overhang is published per row as `native_overhang_h`.
  - §2.5: documents that the same `lead_day` is a different forecast range at different longitudes (up to 6 h across the CONUS), and that the `ALL` row mixes them.
  - §2.6 / §1.1: station metadata gains `grid_elev_m` and `dz_m`; the station-selection rule is stated, together with what it does to external validity.
  - §4: the persistence baseline is now the **same functional** as the numerator (instant vs instant, sampled extreme vs sampled extreme), which removes the sampling penalty from the skill ratio. `n_common` and `mae_persistence_common` are published so that the skill column is recomputable from its own row. `mae_debiased` becomes **out of sample** (trailing 30 scored days, minimum 15) with a new `n_debiased`. The `ALL` rule is restated as "each functional averaged separately".
  - §5: the day bootstrap is drawn **per group on that group's own realised dates**, seeded from the date set, replacing the shared-resample-matrix approximation; no interval below 28 days or 4 blocks. Hit rates move to **Wilson** intervals. `skill_ci_low/high` added. `significant` renamed `distinguishable_uncorrected`; new `p_boot` and `distinguishable_holm` (Holm over the comparisons against the leader within one displayed table), and only the latter is marked on the site.
  - §10: rewritten; adds the unattributed 18/00 UTC AI cold-bias observation with its three candidate explanations and the planned attribution analyses, and no causal claim.
- **0.2 (2026-08-30)** — methodology and statistics review.
  - §2.4 native extremes: the "buckets entirely inside the day" rule (which yielded values only at −6 h stations) is replaced by "the contiguous overlapping run, at most one crossing bucket at each end, overhang ≤ 6 h"; the overhang is documented per station offset. Native extremes remain diagnostic.
  - §3: preliminary CLI reports are dropped rather than ranked below the observation fallback. NWS rounding (nearest °F, mid-points up; ASOS internal tenths of °C) documented; hit-rate thresholds stated as inclusive.
  - §4: new published columns `n_stations`, `n_flagged`, `mae_debiased`, `skill_persistence_debiased`, `rmse_ci_low/high`, `hit1f_ci_low/high`, `model_version`, `segment_start`. The lagged-persistence baseline and the day-mean `ALL` aggregate are stated as rulings with their rejected alternatives.
  - §5: the day bootstrap is now a **circular moving-block** bootstrap (block 7 days). The shared-resample-matrix approximation is documented and measured against a textbook per-group bootstrap.
  - §7: version segmentation is implemented, not merely asserted — the scoring window is truncated to the latest `model_version` segment.
- **0.1 (2026-08-30)** — initial pre-release methodology.
