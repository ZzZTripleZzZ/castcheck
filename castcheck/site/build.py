"""Static site generator (DESIGN §6).

Jinja2 → ``public/``.  Every page is readable with JavaScript disabled: the tables *are* the
content, every figure is server-rendered inline SVG (``site/svg.py``) with an equivalent table
beside it, and ``assets/chart.js`` only adds the theme toggle and a hover read-out.

Every analysis choice is a URL, not a widget.  The four dimensions — window (30d/90d/365d/all),
initialization (00Z/12Z), interpolation (bilinear/nearest) and variable (t2/tmax_s/tmin_s) — are
baked into static paths so that any view can be linked, cited and diffed:

``/``                                           the default view: 90d · 00Z · bilinear · t2
``/v/{window}-{init}z-{method}-{variable}/``     each of the 48 leaderboard views (``/`` is a copy)
``/station/{ICAO}/``  ``/station/{ICAO}/v/{window}-{init}z-{method}/``
``/model/{model_id}/``  ``/model/{model_id}/v/{window}-{init}z-{method}/``
``/station/{ICAO}/model/{model_id}/lead/{d}/``  the permanent link (DESIGN §6, fixed for good)
``/methodology/``  ``/status/``  ``/data/``  ``/api/v1/``  ``/feed.xml``  ``/_headers``

``station_id="ALL"`` is published as a pseudo-station so the cross-station aggregate has permanent
links too.  Errors are stored in °C (METHODOLOGY §3) and displayed in °F throughout.

Headline variable (METHODOLOGY v0.3, DESIGN §10.2): ``t2`` — the instantaneous 2 m temperature at
the four common synoptic instants, verified against the observation at the same instant.  The
like-for-like daily extremes ``tmax_s``/``tmin_s`` (max/min of the four *observed* samples) are the
second table.  ``tmax_cli``/``tmin_cli`` — the four forecast samples against the NWS daily extremes
— are secondary only: they carry a sampling penalty whose size depends on each model's own diurnal
amplitude, so they are published but never ranked.

Every column the templates read is optional: a scores table written by an older methodology version
simply renders "—" where a column is missing, so the site builds during a methodology transition.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import re
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .. import METHODOLOGY_VERSION, SCHEMA_VERSION, __version__
from ..api import (
    DIAGNOSTICS_INIT,
    DIAGNOSTICS_LEAD,
    DIAGNOSTICS_METHOD,
    DIAGNOSTICS_PENALTY_PAIRS,
    DIAGNOSTICS_WINDOW,
    SKILL_MIN_COMMON,
    diagnostics_payload,
)
from ..config import (
    PUBLIC_DIR,
    REPO_ROOT,
    ModelSpec,
    Station,
    display_names,
    load_models,
    load_stations,
)
from ..store import DAILY_COLUMNS, TRUTH_COLUMNS
from ..verify import (
    ALL_STATIONS,
    MIN_N,
    PAIRWISE_COLUMNS,
    PERSISTENCE_ID,
    SCORE_COLUMNS,
    error_table,
    select_truth,
)
from . import svg

#: DESIGN §10.1.  Read from the store once M1a publishes it, so the /data/ schema table cannot
#: drift from the table it documents.
try:
    from ..store import TRUTH_INSTANT_COLUMNS as _TRUTH_INSTANT_COLUMNS
except ImportError:  # pragma: no cover - transitional, until truth_instant lands
    _TRUTH_INSTANT_COLUMNS = [
        "station_id", "valid_time", "temp_c", "obs_time", "source", "n_reports", "qc_flag",
        "schema_version", "methodology_version",
    ]

__all__ = [
    "CLI_VARIABLES",
    "COLUMN_DOCS",
    "COLUMN_DOCS_BY_TABLE",
    "CONCEPT_DOI",
    "FAIRNESS",
    "FAIRNESS_BANNER",
    "HEADLINE_VARIABLE",
    "SAMPLED_VARIABLES",
    "SITE_URL",
    "VIEWS",
    "build_site",
    "changelog_entries",
    "citation",
    "citation_long",
    "human_time",
    "minify_html",
    "next_update",
    "source_commit",
    "view_slug",
]

log = logging.getLogger("castcheck.site")

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
ASSETS = HERE / "assets"

SITE_URL = "https://castcheck.zifanzhang.com"
REPO_URL = "https://github.com/ZzZTripleZzZ/castcheck"
#: Zenodo *concept* DOI: it always resolves to the newest deposited version, which is what a
#: citation of the project as a whole should point at.  Each release also has its own version DOI;
#: the long citation and the dataset citation carry the concept DOI, the one-line short citation
#: stays one line and carries the permanent URL only.
CONCEPT_DOI = "10.5281/zenodo.22212363"
HF_URL = "https://huggingface.co/datasets/castcheck/temperature-verification"
HF_FILES_URL = f"{HF_URL}/tree/main"

#: METHODOLOGY §7 in one line — the banner on every page, which must not cost two lines of
#: screen at desktop width. The full statement is the methodology section it links to.
FAIRNESS_BANNER = (
    "Raw model output on the native 0.25° grid — no MOS, no bias correction, no post-processing. "
    "Scores understate operational forecast quality."
)

#: METHODOLOGY §7 in full; quoted in the API and the data page.
FAIRNESS = (
    "These are raw model outputs on the native 0.25° grid, without MOS, bias correction, "
    "downscaling or any post-processing. They are not equivalent to the products end users receive "
    "from a weather service or app, and the scores here understate operational forecast quality."
)

HEADLINE_LEADS = (1, 3, 5, 7)
SPARK_LEADS = tuple(range(1, 10))
WINDOWS = ("30d", "90d", "365d", "all")
INITS = (0, 12)
METHODS = ("bilinear", "nearest")

#: DESIGN §10.2.  ``t2`` is the headline: the instantaneous value at the four common instants,
#: against the observation at the same instant, so nothing about the metric depends on a model's
#: own diurnal amplitude.
HEADLINE_VARIABLE = "t2"
#: One curve per synoptic hour — diurnal structure, published on the permanent links only.
HOUR_VARIABLES = ("t2_00z", "t2_06z", "t2_12z", "t2_18z")
#: Like-for-like daily extremes: forecast samples against the *observed* samples.
SAMPLED_VARIABLES = ("tmax_s", "tmin_s")
#: Secondary: the same four forecast samples against the NWS daily extremes.  Never ranked.
CLI_VARIABLES = ("tmax_cli", "tmin_cli", "tmax_native_cli", "tmin_native_cli")
#: The variables the switcher offers and the leaderboard views are built for.
VARIABLES = (HEADLINE_VARIABLE, *SAMPLED_VARIABLES)
#: Everything a station or model page shows above the secondary block.
PAGE_VARIABLES = (HEADLINE_VARIABLE, *SAMPLED_VARIABLES)
#: Scores written before methodology v0.3 used these names for what is now ``*_cli``.
LEGACY_VARIABLE_ALIAS = {"tmax": "tmax_cli", "tmin": "tmin_cli"}
DEFAULT_WINDOW = "90d"
DEFAULT_INIT = 0
DEFAULT_METHOD = "bilinear"
DEFAULT_VARIABLE = HEADLINE_VARIABLE
DEFAULT_VIEW = (DEFAULT_WINDOW, DEFAULT_INIT, DEFAULT_METHOD, DEFAULT_VARIABLE)
#: every leaderboard combination, each one a static page
VIEWS = tuple((w, i, m, v) for w in WINDOWS for i in INITS for m in METHODS for v in VARIABLES)
#: Station and model pages carry both variables at once and are always bilinear: they are
#: navigation, and the interpolation sensitivity belongs where the numbers are cited — every
#: permanent link publishes all 32 window × init × method × variable combinations on one page.
SUBVIEWS = tuple((w, i) for w in WINDOWS for i in INITS)

C_TO_F_DELTA = 9.0 / 5.0
RECENT_TRUTH_DAYS = 30
SERIES_DAYS = 90
WINDOW_ORDER = {"30d": 0, "90d": 1, "365d": 2, "all": 3}
WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "all": None}
PUBLISH_HOUR_UTC = 11  # verify-publish.yml (DESIGN §7)
#: The reference model of the bias map: fixed, never the per-station winner (review A6).
REFERENCE_MODEL = "ifs_hres"
#: Below this the bootstrap does not produce an interval (DESIGN §10.3); the site shows "—".
CI_MIN_N = 28
CI_MIN_BLOCKS = 4

VAR_LABEL = {
    "t2": "instantaneous temperature",
    "t2_00z": "instantaneous temperature, 00Z",
    "t2_06z": "instantaneous temperature, 06Z",
    "t2_12z": "instantaneous temperature, 12Z",
    "t2_18z": "instantaneous temperature, 18Z",
    "tmax_s": "sampled daily maximum",
    "tmin_s": "sampled daily minimum",
    "tmax_cli": "daily maximum vs NWS CLI",
    "tmin_cli": "daily minimum vs NWS CLI",
    "tmax_native_cli": "native daily maximum vs NWS CLI",
    "tmin_native_cli": "native daily minimum vs NWS CLI",
    # pre-v0.3 names, kept so an old scores table still renders a label
    "tmax": "daily maximum vs NWS CLI",
    "tmin": "daily minimum vs NWS CLI",
}
VAR_SHORT = {
    "t2": "t2 (instantaneous)",
    "tmax_s": "sampled Tmax",
    "tmin_s": "sampled Tmin",
    "tmax_cli": "Tmax vs CLI",
    "tmin_cli": "Tmin vs CLI",
    "tmax_native_cli": "native Tmax vs CLI",
    "tmin_native_cli": "native Tmin vs CLI",
}
VAR_TRUTH = {
    "t2": "the observed 2 m temperature at the same instant (ASOS routine METAR)",
    "tmax_s": "the maximum of the four *observed* samples on the same day",
    "tmin_s": "the minimum of the four *observed* samples on the same day",
    "tmax_cli": "the NWS Daily Climate Report maximum",
    "tmin_cli": "the NWS Daily Climate Report minimum",
    "tmax_native_cli": "the NWS Daily Climate Report maximum",
    "tmin_native_cli": "the NWS Daily Climate Report minimum",
}

#: The fixed sentence that must accompany every CLI-truth number (review A2, DESIGN §10.2).
CLI_CAVEAT = (
    "This comparison scores four 6-hourly forecast samples against the true daily extreme, so it "
    "includes a sampling penalty. The size of that penalty depends on each model's own diurnal "
    "amplitude, which is one of the things this site measures, so these numbers are published for "
    "operational relevance and are never used for ranking."
)

#: Columns whose name is shared across tables but whose *meaning* is not.  ``station_id`` carries
#: the ``ALL`` pseudo-station only where an aggregate exists; ``variable``, ``valid_time`` and
#: ``source`` mean different things in the 6-hourly extraction, in the observed instants and in the
#: daily truth.  Looked up before :data:`COLUMN_DOCS`, which holds the shared default.
COLUMN_DOCS_BY_TABLE: dict[str, dict[str, tuple[str, str, str]]] = {
    "forecast_values": {
        "variable": ("string", "—", "the field this value was extracted from: t2 (instantaneous "
                     "2 m temperature, the only one that is scored), or one of the models' native "
                     "extreme fields — mx2t3/mn2t3 and mx2t6/mn2t6 (ECMWF, 3 h and 6 h "
                     "accumulations) and tmax6/tmin6 (GFS, 6 h). The native fields are diagnostic "
                     "only (METHODOLOGY §2.4)"),
    },
    "truth_instant": {
        "valid_time": ("timestamp", "UTC", "the synoptic instant the observation belongs to — one "
                       "of 00/06/12/18 UTC, not a forecast valid time. The timestamp of the report "
                       "actually used is obs_time, within ±35 min of this"),
        "source": ("string", "—", "where the observation came from: ASOS_IEM (the Iowa "
                   "Environmental Mesonet METAR archive) or NWS_API (api.weather.gov, used for "
                   "the last week, before the archive catches up)"),
    },
    "scores": {
        "station_id": ("string", "—", "ICAO identifier, or ALL for the cross-station aggregate"),
    },
    "pairwise": {
        "station_id": ("string", "—", "ICAO identifier, or ALL for the cross-station aggregate"),
    },
}

#: /data/ schema table.  column → (type, unit, meaning)
COLUMN_DOCS: dict[str, tuple[str, str, str]] = {
    "station_id": ("string", "—", "ICAO identifier of the station the row belongs to"),
    "model_id": ("string", "—", "stable model identifier from config/models.yaml"),
    "model_version": ("string", "—", "upstream cycle/version string as advertised by the producer"),
    "init_hour": ("int8", "UTC hour", "model initialization hour, 0 or 12"),
    "init_time": ("timestamp", "UTC", "model initialization time"),
    "valid_time": ("timestamp", "UTC", "forecast valid time"),
    "lead_h": ("int16", "hours", "valid_time − init_time"),
    "lead_day": ("int8", "days", "target climatological date − UTC date of the initialization"),
    "variable": ("string", "—", "t2 (instantaneous, headline), t2_00z…t2_18z (one synoptic hour), "
                 "tmax_s/tmin_s (extremes of the four samples vs the extremes of the four observed "
                 "samples), tmax_cli/tmin_cli (the same forecast samples vs the NWS daily extreme, "
                 "secondary) or tmax_native_cli/tmin_native_cli (native extreme fields vs the NWS "
                 "daily extreme, secondary)"),
    "bucket_h": ("int8", "hours", "accumulation window of a native extreme field, 0 if instantaneous"),
    "method": ("string", "—", "grid-to-station interpolation: bilinear (headline) or nearest"),
    "window": ("string", "—", "scoring window: 30d, 90d, 365d or all"),
    "n": ("int32", "days", "number of scored climatological days in the window"),
    "n_stations": ("float32", "stations", "mean number of stations behind each day of an ALL row "
                   "(1 for a single-station row)"),
    "n_flagged": ("int32", "days", "how many of those days carry a QC flag on the observation"),
    "n_common": ("int32", "days", "days used as the denominator of the skill score: days on which "
                 "both the model and the persistence baseline have a value (in the pairwise table, "
                 "days on which both models of the pair have a forecast)"),
    "n_samples": ("int8", "count", "how many of the four common samples were present (0–4)"),
    "valid_hour_utc": ("int8", "UTC hour", "which of the four common instants a row belongs to; 0 "
                       "for the one-row-per-day variables"),
    "n_debiased": ("int32", "days", "days that had at least 15 of the preceding 30 scored days "
                   "available to estimate the out-of-sample bias correction"),
    "mae": ("float32", "°C", "mean absolute error"),
    "bias": ("float32", "°C", "mean signed error; positive = model too warm"),
    "rmse": ("float32", "°C", "root mean squared error"),
    "hit1f": ("float32", "fraction", "share of days with |error| ≤ 1 °F"),
    "hit2f": ("float32", "fraction", "share of days with |error| ≤ 2 °F"),
    "hit3f": ("float32", "fraction", "share of days with |error| ≤ 3 °F"),
    "mae_debiased": ("float32", "°C", "MAE after removing an out-of-sample bias estimate: the bias "
                     "of the trailing 30 scored days before each day (minimum 15) applied forward "
                     "to that day. Days without enough history are excluded"),
    "skill_persistence": ("float32", "fraction",
                          "1 − MAE ÷ mae_persistence_common, both computed on the same n_common "
                          "days; positive is better"),
    "mae_persistence_common": ("float32", "°C", "MAE of the persistence baseline restricted to the "
                               "n_common days — the denominator of skill_persistence"),
    "skill_ci_low": ("float32", "fraction", "2.5th percentile of the paired skill bootstrap"),
    "skill_ci_high": ("float32", "fraction", "97.5th percentile of the paired skill bootstrap"),
    "skill_persistence_debiased": ("float32", "fraction",
                                   "the same skill score computed on the out-of-sample debiased "
                                   "errors, so a station with a large constant offset is not "
                                   "scored as skill-less"),
    "mae_ci_low": ("float32", "°C", "2.5th percentile of the per-group moving-block bootstrap; "
                   "empty when n < 28 or fewer than 4 blocks"),
    "mae_ci_high": ("float32", "°C", "97.5th percentile of the per-group moving-block bootstrap; "
                    "empty when n < 28 or fewer than 4 blocks"),
    "bias_ci_low": ("float32", "°C", "2.5th percentile of the bias bootstrap"),
    "bias_ci_high": ("float32", "°C", "97.5th percentile of the bias bootstrap"),
    "rmse_ci_low": ("float32", "°C", "2.5th percentile of the RMSE bootstrap"),
    "rmse_ci_high": ("float32", "°C", "97.5th percentile of the RMSE bootstrap"),
    "hit1f_ci_low": ("float32", "fraction", "lower bound of the Wilson score 95 % interval of the "
                     "±1 °F hit rate"),
    "hit1f_ci_high": ("float32", "fraction", "upper bound of the Wilson score 95 % interval of the "
                      "±1 °F hit rate"),
    "segment_start": ("date", "LST day", "first day of the current model-version segment; scores "
                      "cover only this segment"),
    "mae_diff": ("float32", "°C", "MAE(model_a) − MAE(model_b) on their common days"),
    "ci_low": ("float32", "°C", "2.5th percentile of the paired bootstrap difference"),
    "ci_high": ("float32", "°C", "97.5th percentile of the paired bootstrap difference"),
    "distinguishable_uncorrected": ("bool", "—", "true when the single 95 % interval of the "
                                    "difference excludes zero, with no correction for the number "
                                    "of comparisons made"),
    "p_boot": ("float32", "—", "two-sided bootstrap p-value of the paired MAE difference"),
    "distinguishable_holm": ("bool", "—", "true after a Holm correction over the family of "
                             "comparisons against the leader within one displayed table (same "
                             "station, initialization, lead, variable, method and window). This is "
                             "the only flag the site marks with ▼ or ▲"),
    "significant": ("bool", "—", "pre-v0.3 name of distinguishable_uncorrected"),
    "model_a": ("string", "—", "first model of the pair"),
    "model_b": ("string", "—", "second model of the pair"),
    "period_start": ("date", "LST day", "first climatological day contributing to the window"),
    "period_end": ("date", "LST day", "last climatological day contributing to the window"),
    "computed_at": ("timestamp", "UTC", "when this table was computed"),
    "climo_date": ("date", "LST day", "climatological day, midnight to midnight local standard time"),
    "source": ("string", "—", "truth product: CLI, CF6 or OBS"),
    "tmax_f": ("int16", "°F", "daily maximum as reported by the NWS (whole degrees)"),
    "tmin_f": ("int16", "°F", "daily minimum as reported by the NWS (whole degrees)"),
    "tmax_c": ("float32", "°C", "daily maximum converted to °C"),
    "tmin_c": ("float32", "°C", "daily minimum converted to °C"),
    "issuance_time": ("timestamp", "UTC", "issuance time of the truth product"),
    "is_final": ("bool", "—", "first CLI issued after local midnight (the first-final policy)"),
    "revised": ("bool", "—", "a later corrected report exists (never used in scores)"),
    "revised_tmax_f": ("int16", "°F", "latest corrected maximum, published but not scored"),
    "revised_tmin_f": ("int16", "°F", "latest corrected minimum, published but not scored"),
    "qc_flag": ("string", "—", "quality note, empty when clean"),
    "product_id": ("string", "—", "api.weather.gov product id or IEM archive key"),
    "value_c": ("float32", "°C", "extracted station value, NaN when missing"),
    "missing_reason": ("string", "—", "why a value is absent; empty when present"),
    "source_url": ("string", "—", "exact object or URL the value was read from"),
    "fetched_at": ("timestamp", "UTC", "when the value was fetched"),
    "tmax_sampled_c": ("float32", "°C", "max of the four common samples in the climatological day"),
    "tmin_sampled_c": ("float32", "°C", "min of the four common samples in the climatological day"),
    "tmax_native_c": ("float32", "°C", "daily max from the model's native extreme field (diagnostic)"),
    "tmin_native_c": ("float32", "°C", "daily min from the model's native extreme field (diagnostic)"),
    "tmax_obs_s_c": ("float32", "°C", "max of the four *observed* instants of the same "
                     "climatological day — the truth tmax_s is scored against, so that forecast "
                     "and observation are sampled identically (METHODOLOGY §2.3)"),
    "tmin_obs_s_c": ("float32", "°C", "min of the four observed instants of the same "
                     "climatological day — the truth tmin_s is scored against"),
    "n_obs_samples": ("int8", "count", "how many of the four observed instants were present (0–4); "
                      "tmax_s/tmin_s are only scored on a day with all four"),
    "native_overhang_h": ("int8", "hours", "how far past the end of the climatological day the "
                          "native extreme accumulation window reaches, 0–6; it is why the native "
                          "comparison is diagnostic only (METHODOLOGY §2.4)"),
    "temp_c": ("float32", "°C", "observed 2 m air temperature at the synoptic instant; empty when "
               "no usable report was found"),
    "obs_time": ("timestamp", "UTC", "timestamp of the report actually used"),
    "n_reports": ("int8", "count", "routine reports found inside the ±35 min window"),
    "grid_elev_m": ("float32", "m", "mean elevation of the 0.25° cell containing the station "
                    "(ETOPO 2022, 60 arc-second, public domain)"),
    "dz_m": ("float32", "m", "elev_m − grid_elev_m: how much higher the station sits than the "
             "model's idea of the ground under it"),
    "market_city": ("string", "—", "the temperature-contract city this station is the settlement "
                    "site for; this is the station-selection rule, not a data source"),
    "iem_id": ("string", "—", "identifier of the station in the IEM ASOS archive"),
    "schema_version": ("string", "—", "data-model version (DESIGN §3)"),
    "methodology_version": ("string", "—", "METHODOLOGY version that produced the numbers"),
}

#: Used only when ``METHODOLOGY.md`` cannot be read; the real changelog is that document's own
#: ``## Changelog`` section, rendered by :func:`changelog_entries` so the two can never diverge.
CHANGELOG_FALLBACK = [
    {"version": "0.1", "date": "2026-08-30", "summary": "Initial pre-release methodology.",
     "details": []},
]

LIMITATIONS = [
    "The headline variable is the instantaneous temperature at four synoptic instants, so it says "
    "nothing about the daily maximum a person experiences. The like-for-like daily extremes "
    "(tmax_s/tmin_s) are the max/min of the same four samples on both sides; the comparison "
    "against the true NWS daily extremes (tmax_cli/tmin_cli) additionally carries a sampling "
    "penalty whose size depends on each model's own diurnal amplitude, which is why it is "
    "published as secondary and never ranked (METHODOLOGY §2.3).",
    "No elevation or lapse-rate correction is applied; stations whose elevation differs sharply "
    "from the 0.25° grid cell carry a representativeness error that is charged to the model. "
    "The size of the offset is published per station on /stations/ as dz_m and a first-order "
    "lapse-rate magnitude.",
    "Truth is the first final NWS CLI report for the CLI variables and the routine METAR nearest "
    "each synoptic hour for the instantaneous ones. Later corrections are stored but never change "
    "a published score, so a corrected observation leaves a permanent, documented discrepancy; "
    "the size of that effect is quantified on /data/.",
    "Models enter the record on different dates, so their windows are not identical. Pairwise "
    "comparisons are computed on common days only; the leaderboard columns are not. The skill "
    "column is the one leaderboard column that is a common-day comparison, and it names its own "
    "denominator and n.",
    "Groups with fewer than 30 scored days are published but greyed out and excluded from every "
    "ranking, and no confidence interval is computed below 28 days or 4 bootstrap blocks; early "
    "in a model's record most windows are in that state.",
    "The same lead day covers different forecast hours at different longitudes: a climatological "
    "day begins at local midnight, so at the easternmost station (UTC−5) a lead day starts six "
    "hours earlier in forecast time than at the westernmost (UTC−8). Every pooled ALL row mixes "
    "forecast ranges that differ by up to six hours across the station set (§2.5).",
    "The 0.25° AIWP archive is a research product whose GFS-initialized models are produced on "
    "alternating cycles; an initialization the archive never produced is marked as such on "
    "/status/ and is not counted as downtime.",
]

#: A publication snapshot is named for its date and nothing else; anything else in the history
#: directory (a sync conflict copy, an editor backup) is not a publication date.
_HISTORY_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

_CHANGELOG_HEAD = re.compile(r"^-\s+\*\*(?P<v>[0-9][^ ]*)\s*\((?P<d>[^)]*)\)\*\*\s*[—-]?\s*(?P<s>.*)$")


def changelog_entries(text: str | None = None) -> list[dict]:
    """The ``## Changelog`` section of METHODOLOGY.md, parsed into rows for /data/.

    The changelog is written once, in the document that the version number belongs to; the site
    renders that section rather than keeping a second copy that can silently fall behind (A5).
    """
    if text is None:
        path = REPO_ROOT / "METHODOLOGY.md"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
    body = text.split("\n## Changelog", 1)
    if len(body) < 2:
        return list(CHANGELOG_FALLBACK)
    out: list[dict] = []
    for raw in body[1].splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            break
        m = _CHANGELOG_HEAD.match(line.strip()) if not line.startswith((" ", "\t")) else None
        if m:
            out.append({"version": m.group("v"), "date": m.group("d").strip(),
                        "summary": m.group("s").strip(), "details": []})
        elif out and line.strip().startswith(("-", "*")) and line.startswith((" ", "\t")):
            out[-1]["details"].append(line.strip().lstrip("-* ").strip())
    return out or list(CHANGELOG_FALLBACK)


def source_commit() -> str:
    """Short hash of the commit that produced this build, or ``local`` outside a git checkout."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "local"
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else "local"


# ------------------------------------------------------------------------------------------
# formatting helpers
# ------------------------------------------------------------------------------------------

def _isnan(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _f(c) -> float | None:
    """A °C *difference* as a float in °F, or None."""
    return None if _isnan(c) else float(c) * C_TO_F_DELTA


#: A displayed negative uses the typographic minus, which is the width of a digit in the mono
#: face and therefore keeps a column of signed numbers aligned. Downloads and the API keep the
#: ASCII hyphen: this substitution is display-only and happens nowhere else.
MINUS = "\u2212"


def _minus(text: str) -> str:
    return text.replace("-", MINUS)


def f_delta(c, digits: int = 2) -> str:
    return "—" if _isnan(c) else _minus(f"{float(c) * C_TO_F_DELTA:.{digits}f}")


def f_signed(c, digits: int = 2) -> str:
    return "—" if _isnan(c) else _minus(f"{float(c) * C_TO_F_DELTA:+.{digits}f}")


def f_ci(lo, hi, digits: int = 2) -> str:
    """The compact interval notation of docs/05 §A: ``[1.9, 2.3]``."""
    if _isnan(lo) or _isnan(hi):
        return "—"
    return _minus(
        f"[{float(lo) * C_TO_F_DELTA:.{digits}f}, {float(hi) * C_TO_F_DELTA:.{digits}f}]")


def f_pct(x) -> str:
    return "—" if _isnan(x) else f"{float(x) * 100:.0f}%"


def f_skill(x) -> str:
    return "—" if _isnan(x) else _minus(f"{float(x):+.2f}")


def f_skill_ci(lo, hi) -> str:
    """A skill interval: a ratio, so it stays unitless (no °F conversion)."""
    if _isnan(lo) or _isnan(hi):
        return "—"
    return _minus(f"[{float(lo):+.2f}, {float(hi):+.2f}]")


def f_int(x) -> str:
    return "—" if _isnan(x) else f"{int(float(x))}"


def f_period(start, end) -> str:
    if _isnan(start) or _isnan(end):
        return "—"
    return f"{start} → {end}"


def permalink_url(station_id: str, model_id: str, lead: int) -> str:
    return f"/station/{station_id}/model/{model_id}/lead/{int(lead)}/"


def citation(station_id: str, model_id: str, lead: int, accessed: str) -> str:
    """The short citation (one line, fits in a footnote)."""
    return (
        f"CastCheck, {station_id} {model_id} lead {int(lead)}, "
        f"methodology v{METHODOLOGY_VERSION}, accessed {accessed}, "
        f"{SITE_URL}{permalink_url(station_id, model_id, lead)}"
    )


def citation_long(station_id: str, station_name: str, model_id: str, lead: int,
                  data_through: str, accessed: str) -> str:
    """The long citation: everything needed to reproduce the number."""
    return (
        f"CastCheck (2026). Station-level verification of raw weather-model 2 m temperature "
        f"forecasts: {station_id} ({station_name}), model {model_id}, lead day {int(lead)}. "
        f"Methodology version {METHODOLOGY_VERSION}, schema version {SCHEMA_VERSION}, "
        f"data through {data_through}. Accessed {accessed}. "
        f"{SITE_URL}{permalink_url(station_id, model_id, lead)} "
        f"doi:{CONCEPT_DOI}"
    )


def view_slug(window: str, init_hour: int, method: str | None = None,
              variable: str | None = None) -> str:
    """``90d-00z`` (station/model view) or ``90d-00z-bilinear-tmax`` (leaderboard view)."""
    base = f"{window}-{int(init_hour):02d}z"
    if method is None:
        return base
    base = f"{base}-{method}"
    return base if variable is None else f"{base}-{variable}"


def next_update(from_iso: str) -> str:
    """The next scheduled publish (DESIGN §7: verify-publish runs at 11:00 UTC daily)."""
    try:
        t = datetime.fromisoformat(from_iso)
    except ValueError:  # pragma: no cover - defensive
        t = datetime.now(UTC)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    nxt = t.replace(hour=PUBLISH_HOUR_UTC, minute=0, second=0, microsecond=0)
    if nxt <= t:
        nxt += timedelta(days=1)
    return nxt.isoformat()


def human_time(iso: str | None) -> str:
    """``"2026-08-31T05:46:12+00:00"`` -> ``"2026-08-31 05:46 UTC"``.

    The machine-readable form stays in the ``datetime`` attribute of a ``<time>`` element, so the
    page is still parseable; what a reader sees is a clock time with a zone, not an RFC 3339
    string with a ``+00:00`` tail they have to decode.
    """
    if not iso:
        return "—"
    try:
        t = datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return t.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _human_size(n: int) -> str:
    x = float(n)
    for unit in ("B", "kB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


# ------------------------------------------------------------------------------------------
# view rows
# ------------------------------------------------------------------------------------------

def _mname(model_idx: dict | None, model_id: str) -> str:
    """The human label for a model id; the id itself is the fallback and the subtitle."""
    return (model_idx or {}).get(model_id, {}).get("display") or model_id


#: ``svg.bias_class`` name -> the one- or two-character alias used by the compact tables (`.bb`).
_BIAS_SHORT = {"is-warm-3": "w3", "is-warm-2": "w2", "is-warm-1": "w1", "is-cool-1": "k1",
               "is-cool-2": "k2", "is-cool-3": "k3", "is-flat": "f0", "is-null": "z0"}


def _row_view(r, station_id: str, model_id: str, lead: int, model_idx: dict | None = None) -> dict:
    bias_f = _f(r["bias"])
    bias_sig = not (_isnan(r["bias_ci_low"]) or _isnan(r["bias_ci_high"])) and not (
        float(r["bias_ci_low"]) <= 0.0 <= float(r["bias_ci_high"])
    )
    n_stations = r.get("n_stations")
    n_flagged = r.get("n_flagged")
    seg = r.get("segment_start")
    n_common = r.get("n_common")
    mae_pers = r.get("mae_persistence_common")
    # A1: the skill denominator is the *intersection* of the model's days and the baseline's, so
    # the persistence row's own n never reproduces it. Both are written next to the number.
    skill_vs = "—"
    if not _isnan(mae_pers):
        skill_vs = f"vs {f_delta(mae_pers)}"
        if not _isnan(n_common):
            skill_vs += f" (n={int(float(n_common))})"
    elif not _isnan(n_common):
        skill_vs = f"n={int(float(n_common))}"
    # A skill score is a ratio of two MAEs over the *intersection* of the two records.  On a
    # handful of common days that ratio moves several tenths when one day changes, which is not a
    # number to publish as if it meant something: below SKILL_MIN_COMMON the site prints "—" and
    # leaves the sample size visible next to it, so the reader sees why.  The value itself stays
    # in the JSON, flagged `skill_reliable: false`.
    skill_reliable = (not _isnan(n_common)) and int(float(n_common)) >= SKILL_MIN_COMMON
    variable = r["variable"]
    return {
        "model_name": _mname(model_idx, model_id),
        "mae_debiased": f_delta(r.get("mae_debiased")),
        "skill_debiased": (f_skill(r.get("skill_persistence_debiased"))
                           if skill_reliable else "—"),
        "skill_ci": (f_skill_ci(r.get("skill_ci_low"), r.get("skill_ci_high"))
                     if skill_reliable else "—"),
        "skill_reliable": skill_reliable,
        "skill_min_common": SKILL_MIN_COMMON,
        "skill_vs": skill_vs,
        "n_common": f_int(n_common),
        "n_debiased": f_int(r.get("n_debiased")),
        "rmse_ci": f_ci(r.get("rmse_ci_low"), r.get("rmse_ci_high")),
        "hit1f_ci": ("—" if _isnan(r.get("hit1f_ci_low")) or _isnan(r.get("hit1f_ci_high"))
                     else f"[{float(r['hit1f_ci_low']) * 100:.0f}%, "
                          f"{float(r['hit1f_ci_high']) * 100:.0f}%]"),
        "n_stations": ("—" if _isnan(n_stations) else f"{float(n_stations):.1f}".rstrip("0").rstrip(".")),
        "n_flagged": 0 if _isnan(n_flagged) else int(n_flagged),
        "model_version": "" if _isnan(r.get("model_version")) else str(r.get("model_version")),
        "segment_start": "" if _isnan(seg) else str(seg),
        "model_id": model_id,
        "station_id": station_id,
        "lead_day": int(lead),
        "variable": variable,
        "variable_label": VAR_LABEL.get(variable, variable),
        "variable_short": VAR_SHORT.get(variable, variable),
        "secondary": variable in CLI_VARIABLES or variable in LEGACY_VARIABLE_ALIAS,
        "init_hour": f"{int(r['init_hour']):02d}",
        "method": r["method"],
        "window": r["window"],
        "n": int(r["n"]),
        "low_n": int(r["n"]) < MIN_N,
        "mae": f_delta(r["mae"]),
        # The MAE interval was never given a display string, so `{{ r.mae_ci }}` rendered empty
        # everywhere it was used — the permanent-link table and the home page's leader KPI.
        "mae_ci": f_ci(r.get("mae_ci_low"), r.get("mae_ci_high")),
        # `_f` already converts to °F, so the value and its interval are on one scale — the error
        # bars on the home page draw the bar from one and the whiskers from the other.
        "mae_f": _f(r["mae"]),
        "mae_ci_low_f": _f(r.get("mae_ci_low")),
        "mae_ci_high_f": _f(r.get("mae_ci_high")),
        "bias": f_signed(r["bias"]),
        "bias_f": bias_f,
        # Width of the diverging rule under the bias, on the same scale as the colour ramp: full
        # bar at the top step of `_BIAS_STEPS`, so length and hue never disagree.
        "bias_w": 0 if bias_f is None else min(100, round(abs(bias_f) / 2.0 * 100)),
        "bias_ci": f_ci(r["bias_ci_low"], r["bias_ci_high"]),
        "bias_class": svg.bias_class(bias_f, bias_sig),
        "bias_short": _BIAS_SHORT.get(svg.bias_class(bias_f, bias_sig), "z0"),
        "bias_significant": bias_sig,
        "rmse": f_delta(r["rmse"]),
        "hit1f": f_pct(r["hit1f"]),
        "hit2f": f_pct(r["hit2f"]),
        "hit3f": f_pct(r["hit3f"]),
        "skill": f_skill(r["skill_persistence"]) if skill_reliable else "—",
        # the raw skill, because the displayed one carries a typographic minus; `None` when the
        # common sample is too small, so no figure or "best skill" mark can use it either
        "skill_f": _f(r["skill_persistence"]) if skill_reliable else None,
        "period": f_period(r["period_start"], r["period_end"]),
        "permalink": permalink_url(station_id, model_id, lead),
    }


def _variant_links(base: str, current: tuple, dims: tuple[tuple[str, str, list], ...]) -> list[dict]:
    """One switcher group per dimension; each entry is a plain link to another static page."""
    groups = []
    for pos, (key, title, options) in enumerate(dims):
        opts = []
        for value, label in options:
            target = list(current)
            target[pos] = value
            opts.append({
                "href": _view_href(base, tuple(target)),
                "label": label,
                "current": value == current[pos],
            })
        groups.append({"key": key, "title": title, "opts": opts})
    return groups


def _view_href(base: str, view: tuple) -> str:
    if len(view) == 4:
        if tuple(view) == DEFAULT_VIEW:
            return base
        return f"{base}v/{view_slug(view[0], view[1], view[2], view[3])}/"
    if tuple(view) == (DEFAULT_WINDOW, DEFAULT_INIT):
        return base
    return f"{base}v/{view_slug(view[0], view[1])}/"


# Compact labels: the switcher is a segmented control, and "last 365 days" wrapped it on a phone.
_WINDOW_OPTS = [(w, "All" if w == "all" else f"{w[:-1]} d") for w in WINDOWS]
_INIT_OPTS = [(i, f"{i:02d}Z") for i in INITS]
_METHOD_OPTS = [(m, m) for m in METHODS]
_VAR_OPTS = [(v, VAR_SHORT.get(v, v)) for v in VARIABLES]

_DIMS4 = (("window", "Window", _WINDOW_OPTS), ("init", "Initialization", _INIT_OPTS),
          ("method", "Interpolation", _METHOD_OPTS), ("variable", "Variable", _VAR_OPTS))
_DIMS2 = _DIMS4[:2]


# ------------------------------------------------------------------------------------------
# generator
# ------------------------------------------------------------------------------------------

def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


#: Elements whose text is significant: whitespace inside them is never touched.
_PRESERVE = re.compile(r"(?is)(<(?:pre|textarea|script|style)\b[^>]*>.*?</(?:pre|textarea|script|"
                       r"style)>)")
_TAG = re.compile(r"<[^>]*>")
_TAG_NAME = re.compile(r"^</?\s*([a-zA-Z][a-zA-Z0-9]*)")

#: Elements between which whitespace has no rendering effect.  Everything else — ``span``, ``a``,
#: ``i``, ``em``, ``code``, ``time`` … — is inline, where a newline in the source *is* the space
#: between two words, so it is collapsed to one space rather than dropped.
_BLOCK_TAGS = frozenset((
    "html", "head", "body", "meta", "link", "title", "base",
    "main", "header", "footer", "nav", "section", "article", "aside", "div", "p", "hr",
    "blockquote", "figure", "figcaption",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd", "details", "summary",
    "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "colgroup", "col",
    "form", "fieldset", "legend", "select", "option", "optgroup",
))

_HAS_NEWLINE = re.compile(r"\n")


def _elem(tag: str) -> str:
    m = _TAG_NAME.match(tag)
    return m.group(1).lower() if m else ""


def minify_html(html: str) -> str:
    """Strip the generator's indentation without changing what the page renders.

    The permanent-link pages are the bulk of the site — one file per station × model × lead day,
    each carrying a table of every window, initialization, interpolation and variable — and the
    template's indentation is a tenth of their bytes.  Two rules, both rendering-neutral:

    * whitespace that sits **between two block-level tags** (a ``</td>`` and the next ``<td>``, a
      ``</li>`` and the next ``<li>``) is dropped: the HTML layout model ignores it;
    * every other run of newline-plus-indent collapses to a **single space**, because next to an
      inline element that whitespace is the space between two words —
      ``<em>a</em>\n<em>b</em>`` renders "a b", not "ab".

    ``<pre>``, ``<textarea>``, ``<script>`` and ``<style>`` are passed through untouched.
    """
    parts = _PRESERVE.split(html)
    for i in range(0, len(parts), 2):  # even indices are outside the preserved elements
        chunk = parts[i]
        if "\n" not in chunk:
            continue
        out: list[str] = []
        prev_tag = ""
        pos = 0
        for m in _TAG.finditer(chunk):
            text = chunk[pos:m.start()]
            if text and not text.strip() and _HAS_NEWLINE.search(text):
                nxt = _elem(m.group(0))
                drop = _elem(prev_tag) in _BLOCK_TAGS and nxt in _BLOCK_TAGS
                out.append("" if drop else " ")
            elif text:
                out.append(_INDENT_RUN.sub(" ", text))
            out.append(m.group(0))
            prev_tag = m.group(0)
            pos = m.end()
        tail = chunk[pos:]
        out.append(_INDENT_RUN.sub(" ", tail) if tail.strip() else
                   ("" if _elem(prev_tag) in _BLOCK_TAGS else tail))
        parts[i] = "".join(out)
    return "".join(parts)


_INDENT_RUN = re.compile(r"\s*\n\s*")


class _Writer:
    def __init__(self, out: Path):
        self.out = out
        self.n = 0
        self.bytes = 0

    def write(self, relpath: str, html: str, count: bool = True) -> Path:
        path = self.out / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        body = minify_html(html)
        path.write_text(body, encoding="utf-8")
        self.bytes += len(body.encode("utf-8"))
        if count:
            self.n += 1
        return path


def _render_markdown(text: str) -> Markup:
    try:
        import markdown as md

        return Markup(md.markdown(text, extensions=["tables", "toc", "attr_list"]))
    except Exception:  # noqa: BLE001  # pragma: no cover - markdown is a soft dependency
        from markupsafe import escape

        return Markup(f"<pre class='mono'>{escape(text)}</pre>")


def _display_map(model_idx: dict) -> dict[str, str]:
    """``model_id -> human label``, matching :func:`castcheck.config.display_names`.

    Derived from the models this build actually used (so an injected registry works in tests),
    with the real ``config/models.yaml`` filling any gap.
    """
    names: dict[str, str] = {}
    for mid, info in model_idx.items():
        family = info.get("family") or mid
        names[mid] = f"{family} ({info['init_field']} init)" if info.get("init_field") else family
    for mid, label in display_names().items():
        names.setdefault(mid, label)
    return names


def _load_instant_errors(stations, models) -> pd.DataFrame | None:
    """Forecast-minus-observed at the four common instants, for the ``t2`` daily-error chart.

    Derived rather than stored (``derive.instant_errors``), which costs well under a second on the
    whole archive.  Any failure — no ``truth_instant`` yet, an unreadable shard — returns ``None``
    and the site simply falls back to the best variable it does have.
    """
    try:
        from .. import store
        from ..derive import instant_errors

        ti = store.read_truth_instant()
        if ti is None or not len(ti):
            return None
        return instant_errors(store.read_forecast_values(), ti, stations, models)
    except Exception:  # noqa: BLE001 - the chart is not worth failing a build for
        log.warning("instant errors unavailable; the daily-error chart falls back", exc_info=True)
        return None


def _normalise_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Rename pre-v0.3 variable names to what they actually were.

    Before methodology v0.3 the only two variables were the max/min of the four forecast samples
    scored against the NWS daily extremes — exactly today's ``tmax_cli``/``tmin_cli``.  Renaming
    them (only when the new names are absent) keeps an old scores table readable and puts those
    numbers in the secondary block where they belong, instead of under a headline they never were.
    """
    if df is None or len(df) == 0 or "variable" not in df.columns:
        return df
    have = set(df["variable"].unique())
    if not (have & set(LEGACY_VARIABLE_ALIAS)) or (have & set(LEGACY_VARIABLE_ALIAS.values())):
        return df
    out = df.copy()
    out["variable"] = out["variable"].map(lambda v: LEGACY_VARIABLE_ALIAS.get(v, v))
    return out


def _model_index(models: list[ModelSpec]) -> dict[str, dict]:
    idx = {
        m.model_id: {
            "model_id": m.model_id, "family": m.family, "source": m.source, "product": m.product,
            "init_field": m.init_field, "inits": list(m.inits), "step_h": m.step_h,
            "max_h": m.max_h, "native_extremes": list(m.native_extremes), "baseline": False,
        }
        for m in models
    }
    idx[PERSISTENCE_ID] = {
        "model_id": PERSISTENCE_ID, "family": "Persistence (baseline)", "source": "truth",
        "product": "yesterday's observation", "init_field": None, "inits": [0, 12], "step_h": 24,
        "max_h": 240, "native_extremes": [], "baseline": True,
    }
    return idx


def build_site(
    as_of: date | str | None = None,
    out: str | Path | None = None,
    scores: pd.DataFrame | None = None,
    pairwise: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    instant: pd.DataFrame | None = None,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
    status_report: dict | None = None,
    api_written: dict[str, int] | None = None,
    permalinks: bool = True,
    truth_instant: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Render the whole site into ``out`` (default ``public/``).  Returns page counts."""
    out = Path(out) if out is not None else PUBLIC_DIR
    out.mkdir(parents=True, exist_ok=True)
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()

    if scores is None or pairwise is None:
        from .. import store

        s, p = store.read_scores()
        scores = scores if scores is not None else s
        pairwise = pairwise if pairwise is not None else p
    if daily is None:
        from .. import store

        try:
            daily = store.read_daily()
        except Exception:  # noqa: BLE001 - an unreadable shard must not break the build
            daily = pd.DataFrame(columns=DAILY_COLUMNS)
    if truth is None:
        from .. import store

        try:
            truth = store.read_truth()
        except Exception:  # noqa: BLE001 - an unreadable shard must not break the build
            truth = pd.DataFrame(columns=TRUTH_COLUMNS)
    if truth_instant is None:
        from .. import store

        try:
            truth_instant = store.read_truth_instant()
        except Exception:  # noqa: BLE001 - the monthly QC block degrades, the build does not
            truth_instant = pd.DataFrame(columns=_TRUTH_INSTANT_COLUMNS)

    scores = scores if scores is not None else pd.DataFrame(columns=SCORE_COLUMNS)
    pairwise = pairwise if pairwise is not None else pd.DataFrame(columns=PAIRWISE_COLUMNS)
    scores = _normalise_variables(scores)
    pairwise = _normalise_variables(pairwise)

    if status_report is None:
        status_json = out / "api" / "v1" / "status.json"
        if status_json.exists():
            import json

            try:
                status_report = json.loads(status_json.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001  # pragma: no cover - a corrupt report is not fatal
                status_report = None

    if as_of is None:
        if truth is not None and len(truth):
            as_of = pd.to_datetime(truth["climo_date"]).max().date()
        else:
            as_of = datetime.now(UTC).date()
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    elif isinstance(as_of, str):
        as_of = pd.Timestamp(as_of).date()
    as_of_s = as_of.isoformat()

    env = _env()
    w = _Writer(out)
    built_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    data_through = as_of_s
    if scores is not None and len(scores) and "period_end" in scores:
        ends = pd.to_datetime(scores["period_end"], errors="coerce").dropna()
        if len(ends):
            data_through = ends.max().date().isoformat()
    base_ctx = {
        "fairness": FAIRNESS,
        "fairness_banner": FAIRNESS_BANNER,
        "methodology_version": METHODOLOGY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "castcheck_version": __version__,
        "as_of": as_of_s,
        "data_through": data_through,
        "built_at": built_at,
        "built_at_human": human_time(built_at),
        "next_update": next_update(built_at),
        "next_update_human": human_time(next_update(built_at)),
        "min_n": MIN_N,
        "skill_min_common": SKILL_MIN_COMMON,
        "ci_min_n": CI_MIN_N,
        "ci_min_blocks": CI_MIN_BLOCKS,
        "site_url": SITE_URL,
        "repo_url": REPO_URL,
        "hf_url": HF_URL,
        "hf_files_url": HF_FILES_URL,
        "commit": source_commit(),
        "cli_caveat": CLI_CAVEAT,
    }

    # Variant directories are named after the view slugs; if the set of views ever changes, the
    # old pages would linger in public/ and be deployed as stale duplicates. They are cheap to
    # regenerate, so drop them first.
    _prune_variants(out)

    # assets and the Cloudflare Pages header rules
    asset_urls = _write_assets(out)
    base_ctx.update(asset_urls)
    _write_headers(out)

    counts = {"pages": 0, "stations": 0, "models": 0, "permalinks": 0, "leaderboards": 0,
              "months": 0}

    # ---- always-present pages -------------------------------------------------------------
    _write_methodology(env, w, base_ctx)
    _write_404(env, w, base_ctx)
    _write_status(env, w, base_ctx, status_report, _display_map(_model_index(models)))
    _write_indexes(env, w, base_ctx, scores, stations, _model_index(models))
    _write_api_index(env, w, base_ctx, api_written)

    if instant is None:
        instant = _load_instant_errors(stations, models)

    empty = scores is None or len(scores) == 0
    errors = pd.DataFrame()
    if not empty and daily is not None and truth is not None and len(daily) and len(truth):
        try:
            errors = error_table(daily, truth, instant)
        except Exception:  # noqa: BLE001 - a malformed shard must not break the build
            errors = pd.DataFrame()
    downloads = _write_downloads(out, scores, pairwise, errors, stations, models)
    _write_data(env, w, base_ctx, api_written, scores, pairwise, downloads, truth)

    # Both of these render an explicit empty state rather than being skipped, so the routes in the
    # navigation exist from the first build and never 404 while the record is still short.
    _write_diagnostics(env, w, base_ctx, scores, models, _model_index(models), stations)
    month_pages = _monthly_pages(_normalise_variables(errors), truth, truth_instant, daily,
                                 stations, _model_index(models), as_of)
    counts["months"] = _write_monthly(env, w, base_ctx, month_pages)
    monthly_entries = _monthly_feed_entries(month_pages)

    if empty:
        w.write("index.html", env.get_template("empty.html").render(
            **base_ctx,
            heading="CastCheck — no scores yet",
            message="No forecast has been matched to an observation yet, so there is nothing to "
                    "rank. This page will fill in automatically once the first climatological day "
                    "has both a model forecast and its NWS Daily Climate Report.",
        ))
        _write_feed(env, w, base_ctx, monthly_entries or None)
        counts["pages"] = w.n
        return _finish(out, counts)

    sc = scores.copy()
    sc["lead_day"] = sc["lead_day"].astype(int)
    sc["init_hour"] = sc["init_hour"].astype(int)
    sc["n"] = sc["n"].astype(int)
    pw = pairwise.copy()
    if len(pw):
        pw["lead_day"] = pw["lead_day"].astype(int)
        pw["init_hour"] = pw["init_hour"].astype(int)

    model_idx = _model_index(models)
    names = _display_map(model_idx)
    for mid, info in model_idx.items():
        info["display"] = names[mid]
    station_idx = {s.id: s for s in stations}
    leads = sorted(sc["lead_day"].unique().tolist())
    scored_ids = set(sc["station_id"])
    station_links = [{"id": s.id, "name": s.name} for s in stations if s.id in scored_ids]

    errors = _normalise_variables(errors)
    chart_variable = _pick_series_variable(errors)
    series_idx, month_idx, allday_idx = _error_indices(errors, chart_variable)
    base_ctx["chart_variable"] = chart_variable
    base_ctx["chart_variable_label"] = VAR_LABEL.get(chart_variable, chart_variable)

    # ---- leaderboards ---------------------------------------------------------------------
    for view in VIEWS:
        html = _leaderboard_html(env, base_ctx, sc, pw, model_idx, station_links, station_idx,
                                 view, len(stations), leads)
        rel = _view_href("/", view).strip("/")
        w.write(f"{rel}/index.html" if rel else "index.html", html)
        counts["leaderboards"] += 1

    # ---- station pages --------------------------------------------------------------------
    truth_sel = select_truth(truth) if truth is not None and len(truth) else pd.DataFrame()
    all_station = Station(id=ALL_STATIONS, name="All stations (mean of daily station errors)",
                          cli_pil="", tz="UTC", std_offset_h=0, lat=None, lon=None, elev_m=None)
    for sid in sorted(scored_ids):
        st = station_idx.get(sid, all_station if sid == ALL_STATIONS else None)
        if st is None:
            st = Station(id=sid, name=sid, cli_pil="", tz="UTC", std_offset_h=0,
                         lat=None, lon=None, elev_m=None)
        sub = sc[sc["station_id"] == sid]
        recent = _recent_truth(truth_sel, sid)
        avail_all = sub[(sub["window"] == "all") & (sub["variable"] == chart_variable)
                        & (sub["lead_day"] == 1)]
        months = _month_block(month_idx, sid, None)
        for view2 in SUBVIEWS:
            # The month, availability and truth blocks do not depend on the view (they are fixed
            # at lead day 1, 00Z, bilinear, or are pure observations), so they live on the
            # canonical page only instead of being copied into all eight variants.
            canonical_view = tuple(view2) == (DEFAULT_WINDOW, DEFAULT_INIT)
            html = _grid_html(
                env, "station.html", base_ctx, sub, view2, leads, model_idx,
                row_key="model_id", station_fixed=sid, model_fixed=None,
                extra={
                    "station": st, "recent": recent if canonical_view else [],
                    "variants": _variant_links(f"/station/{sid}/", view2, _DIMS2),
                    "availability": _availability(
                        avail_all[(avail_all["init_hour"] == view2[1])
                                  & (avail_all["method"] == DEFAULT_METHOD)], model_idx)[0]
                    if canonical_view else [],
                    "months": months if canonical_view else [],
                    "canonical_view": canonical_view,
                    "canonical_href": f"/station/{sid}/",
                },
            )
            rel = _view_href(f"/station/{sid}/", view2).strip("/")
            w.write(f"{rel}/index.html", html)
        counts["stations"] += 1

    # ---- model pages ----------------------------------------------------------------------
    for mid in sorted(sc["model_id"].unique().tolist()):
        sub = sc[(sc["model_id"] == mid) & (sc["station_id"] != ALL_STATIONS)]
        if sub.empty:
            sub = sc[sc["model_id"] == mid]
        info = dict(model_idx.get(mid, {"model_id": mid, "family": mid, "source": "?",
                                        "product": "?", "init_field": None, "inits": [0, 12],
                                        "step_h": 6, "max_h": 240, "native_extremes": [],
                                        "baseline": False}))
        info.update(_segment_note(sub))
        months = _month_block(month_idx, None, mid)
        for view2 in SUBVIEWS:
            canonical_view = tuple(view2) == (DEFAULT_WINDOW, DEFAULT_INIT)
            html = _grid_html(
                env, "model.html", base_ctx, sub, view2, leads, model_idx,
                row_key="station_id", station_fixed=None, model_fixed=mid,
                extra={
                    "model": info,
                    "variants": _variant_links(f"/model/{mid}/", view2, _DIMS2),
                    "availability": [],
                    "months": months if canonical_view else [],
                    "canonical_view": canonical_view,
                    "canonical_href": f"/model/{mid}/",
                },
            )
            rel = _view_href(f"/model/{mid}/", view2).strip("/")
            w.write(f"{rel}/index.html", html)
        counts["models"] += 1

    # ---- permanent links ------------------------------------------------------------------
    if permalinks:
        counts["permalinks"] = _write_permalinks(
            env, w, out, base_ctx, sc, pw, model_idx, station_idx, series_idx, month_idx,
            allday_idx, as_of_s, data_through)

    _write_feed(env, w, base_ctx,
                _feed_entries(sc, model_idx, as_of_s, built_at) + monthly_entries)
    counts["pages"] = w.n
    return _finish(out, counts)


#: Cloudflare Pages refuses a deployment above these; the build says how close it is.
MAX_FILES = 20_000
MAX_FILE_BYTES = 25 * 1024 * 1024


def _finish(out: Path, counts: dict[str, int]) -> dict[str, int]:
    """Record and log what will be deployed, and warn before a Pages limit is actually hit."""
    n_files, n_bytes = _tree_size(out)
    counts["files"] = n_files
    counts["bytes"] = n_bytes
    log.info("public/: %d files, %s (%d pages)", n_files, _human_size(n_bytes), counts["pages"])
    if n_files > 0.9 * MAX_FILES:
        log.warning("public/ holds %d files; Cloudflare Pages allows %d", n_files, MAX_FILES)
    big = [(p, p.stat().st_size) for p in out.rglob("*")
           if p.is_file() and p.stat().st_size > 0.8 * MAX_FILE_BYTES]
    for p, size in sorted(big, key=lambda t: -t[1]):
        log.warning("%s is %s; Cloudflare Pages refuses files above %s",
                    p.relative_to(out), _human_size(size), _human_size(MAX_FILE_BYTES))
    return counts


# ------------------------------------------------------------------------------------------
# error-series indices (built once, read by every permanent link)
# ------------------------------------------------------------------------------------------

#: Preference order for the one variable the daily-error chart, the month table and the
#: permanent-link CSV are drawn for.  The headline when it exists; otherwise the best available,
#: so a build made during a methodology transition still shows a series instead of an empty frame.
_SERIES_PREFERENCE = (HEADLINE_VARIABLE, *SAMPLED_VARIABLES, *CLI_VARIABLES)


def _pick_series_variable(errors: pd.DataFrame) -> str:
    if errors is None or len(errors) == 0 or "variable" not in errors.columns:
        return DEFAULT_VARIABLE
    have = set(errors["variable"].unique())
    for v in _SERIES_PREFERENCE:
        if v in have:
            return v
    return DEFAULT_VARIABLE


def _error_indices(errors: pd.DataFrame, variable: str = DEFAULT_VARIABLE):
    """``(series, months, allday)`` keyed by ``(station_id, model_id, lead_day)``.

    ``series`` is the last :data:`SERIES_DAYS` of signed daily error in °F for the default
    init/method/variable, ``months`` is every calendar month of the same slice, and ``allday`` is
    the complete signed-error list used for the histogram.  The cross-station ``ALL`` rows are the
    mean over the stations present on each day, exactly as ``verify.score`` aggregates them.
    """
    empty: dict = {}
    if errors is None or len(errors) == 0:
        return empty, empty, empty
    e = errors[
        (errors["init_hour"].astype(int) == DEFAULT_INIT)
        & (errors["method"] == DEFAULT_METHOD)
        & (errors["variable"] == variable)
    ][["station_id", "model_id", "lead_day", "climo_date", "err"]].copy()
    if e.empty:
        return empty, empty, empty
    e["climo_date"] = pd.to_datetime(e["climo_date"])
    e["lead_day"] = e["lead_day"].astype(int)
    agg = (
        e.groupby(["model_id", "lead_day", "climo_date"], observed=True)["err"].mean().reset_index()
    )
    agg["station_id"] = ALL_STATIONS
    e = pd.concat([e, agg[["station_id", "model_id", "lead_day", "climo_date", "err"]]],
                  ignore_index=True)
    e["err_f"] = e["err"].astype(float) * C_TO_F_DELTA
    e["month"] = e["climo_date"].dt.strftime("%Y-%m")
    e = e.sort_values("climo_date")

    # The pooled ``t2`` variable carries four rows per group-day (one per common instant).  The
    # distribution and the monthly MAE are over those values, because that is the population the
    # score is computed on; the time series needs one point per calendar day, so it plots the
    # day mean of the four.
    per_day = (e.groupby(["station_id", "model_id", "lead_day", "climo_date"], observed=True,
                         as_index=False)["err_f"].mean().sort_values("climo_date"))

    series: dict = {}
    allday: dict = {}
    cutoff = e["climo_date"].max() - pd.Timedelta(days=SERIES_DAYS - 1)
    for key, grp in e.groupby(["station_id", "model_id", "lead_day"], observed=True):
        allday[(key[0], key[1], int(key[2]))] = grp["err_f"].tolist()
    for key, grp in per_day.groupby(["station_id", "model_id", "lead_day"], observed=True):
        k = (key[0], key[1], int(key[2]))
        tail = grp[grp["climo_date"] >= cutoff]
        # a continuous calendar axis: days with no score are holes, not joined-up line
        if len(tail):
            idx = pd.date_range(tail["climo_date"].min(), tail["climo_date"].max(), freq="D")
            s = tail.set_index("climo_date")["err_f"].reindex(idx)
            series[k] = {
                "dates": [d.date().isoformat() for d in idx],
                "values": [None if pd.isna(v) else float(v) for v in s],
            }

    months: dict = {}
    mg = e.groupby(["station_id", "model_id", "lead_day", "month"], observed=True)["err_f"]
    stat = mg.agg(n="count", bias="mean")
    stat["mae"] = mg.apply(lambda s: s.abs().mean())
    for (sid, mid, lead, month), r in stat.iterrows():
        months.setdefault((sid, mid, int(lead)), []).append({
            "month": month, "n": int(r["n"]),
            "mae": f"{r['mae']:.2f}", "bias": f"{r['bias']:+.2f}",
            "bias_class": svg.bias_class(float(r["bias"])),
            "bias_short": _BIAS_SHORT.get(svg.bias_class(float(r["bias"])), "z0"),
        })
    return series, months, allday


def _month_block(month_idx: dict, station_id: str | None, model_id: str | None) -> list[dict]:
    """Monthly MAE at lead day 1, either for one station (rows = models) or one model (= stations)."""
    if not month_idx:
        return []
    rows: dict[str, dict[str, dict]] = {}
    keys: set[str] = set()
    for (sid, mid, lead), items in month_idx.items():
        if lead != 1:
            continue
        if station_id is not None and sid != station_id:
            continue
        if model_id is not None and (mid != model_id or sid == ALL_STATIONS):
            continue
        key = mid if station_id is not None else sid
        keys.add(key)
        for it in items:
            rows.setdefault(it["month"], {})[key] = it
    if not rows:
        return []
    order = sorted(keys)
    return [{"month": m, "cells": [rows[m].get(k) for k in order], "cols": order}
            for m in sorted(rows, reverse=True)[:18]]


# ------------------------------------------------------------------------------------------
# page builders
# ------------------------------------------------------------------------------------------

def _leaderboard_html(env, base_ctx, sc, pw, model_idx, station_links, station_idx, view,
                      n_stations, leads):
    window, init_hour, method, variable = view
    sub = sc[
        (sc["station_id"] == ALL_STATIONS)
        & (sc["window"] == window)
        & (sc["init_hour"] == int(init_hour))
        & (sc["method"] == method)
        & (sc["variable"] == variable)
    ]
    pw_sub = pd.DataFrame()
    if len(pw):
        pw_sub = pw[(pw["station_id"] == ALL_STATIONS) & (pw["window"] == window)
                    & (pw["init_hour"] == int(init_hour)) & (pw["method"] == method)
                    & (pw["variable"] == variable)]

    boards = []
    for lead in HEADLINE_LEADS:
        part = sub[sub["lead_day"] == lead]
        if part.empty:
            continue
        boards.append(_board(part, pw_sub, model_idx, lead))

    headline = boards[0] if boards else None
    if headline is None:
        for lead in sorted(sub["lead_day"].unique().tolist()):
            part = sub[sub["lead_day"] == lead]
            if not part.empty:
                headline = _board(part, pw_sub, model_idx, int(lead))
                break

    # The like-for-like daily extremes, always shown under the headline table: the same four
    # samples on the forecast and the observation side, so the sampling definition cancels.
    lead2 = headline["lead"] if headline else HEADLINE_LEADS[0]
    sampled = []
    for var in SAMPLED_VARIABLES:
        part = sc[
            (sc["station_id"] == ALL_STATIONS) & (sc["window"] == window)
            & (sc["init_hour"] == int(init_hour)) & (sc["method"] == method)
            & (sc["variable"] == var) & (sc["lead_day"] == lead2)
        ]
        if part.empty:
            continue
        pw_var = pd.DataFrame()
        if len(pw):
            pw_var = pw[(pw["station_id"] == ALL_STATIONS) & (pw["window"] == window)
                        & (pw["init_hour"] == int(init_hour)) & (pw["method"] == method)
                        & (pw["variable"] == var)]
        board = _board(part, pw_var, model_idx, lead2)
        board["variable"] = var
        board["variable_label"] = _sentence(VAR_LABEL.get(var, var))
        sampled.append(board)

    mae_fig = ""
    if headline:
        bars = [{
            "name": r["model_name"], "mae": r["mae_f"],
            "ci_low": r["mae_ci_low_f"], "ci_high": r["mae_ci_high_f"],
            "best": r["best_mae"], "baseline": r["baseline"], "low_n": r["low_n"],
        } for r in headline["rows"]]
        mae_fig = Markup(svg.mae_bars(
            bars, label=f"MAE by model with 95 % confidence intervals, lead day "
                        f"{headline['lead']}, {window}, {int(init_hour):02d}Z, {method}"))

    matrix = _lead_matrix(sub, model_idx, leads)
    avail = sc[
        (sc["station_id"] == ALL_STATIONS) & (sc["window"] == "all")
        & (sc["variable"] == variable) & (sc["init_hour"] == int(init_hour))
        & (sc["method"] == method) & (sc["lead_day"] == 1)
    ]
    availability, avail_start, avail_end = _availability(avail, model_idx)
    maps = _bias_maps(sc, station_idx, view, model_idx)

    return env.get_template("index.html").render(
        **base_ctx,
        boards=boards,
        sampled=sampled,
        sampled_lead=lead2,
        spark_leads=list(SPARK_LEADS),
        headline=headline,
        mae_fig=mae_fig,
        n_systems=len(model_idx),
        n_views=len(VIEWS),
        canonical_path=_view_href("/", view),
        matrix=matrix,
        window=window,
        window_label="all available history" if window == "all" else f"the last {window[:-1]} days",
        init_hour=f"{int(init_hour):02d}",
        method=method,
        variable=variable,
        variable_label=VAR_LABEL.get(variable, variable),
        variable_truth=VAR_TRUTH.get(variable, "the matching observation"),
        is_headline=variable == HEADLINE_VARIABLE,
        n_stations=n_stations,
        variants=_variant_links("/", view, _DIMS4),
        availability=availability,
        avail_start=avail_start,
        avail_end=avail_end,
        station_links=station_links,
        maps=maps,
        canonical=f"{SITE_URL}{_view_href('/', view)}",
    )


def _board(part: pd.DataFrame, pw_sub: pd.DataFrame, model_idx: dict, lead: int) -> dict:
    part = part.sort_values("mae")
    rows = []
    for _, r in part.iterrows():
        v = _row_view(r, ALL_STATIONS, r["model_id"], lead, model_idx)
        v["baseline"] = model_idx.get(r["model_id"], {}).get("baseline", False)
        rows.append(v)
    ranked = [r for r in rows if not (r["low_n"] or r["baseline"])]
    others = [r for r in rows if r["low_n"] or r["baseline"]]
    leader = ranked[0]["model_id"] if ranked else None
    marks = _significance(pw_sub, lead, leader)
    best_mae = min((r["mae_f"] for r in ranked if r["mae_f"] is not None), default=None)
    best_skill = None
    for r in ranked:
        s = r.get("skill_f")
        if s is None:
            continue
        best_skill = s if best_skill is None else max(best_skill, s)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["mark"] = "★" if r["model_id"] == leader else marks.get(r["model_id"], "·")
        r["mark_title"] = _MARK_TITLE.get(r["mark"], "")
        r["best_mae"] = best_mae is not None and r["mae_f"] is not None and abs(r["mae_f"] - best_mae) < 1e-9
        r["best_skill"] = (best_skill is not None and r["skill_f"] is not None
                           and abs(r["skill_f"] - best_skill) < 1e-9)
    # The skill meter is a within-table comparison: full bar = the best skill on show.
    for r in rows:
        r["skill_w"] = 0
        if r["baseline"] or best_skill is None or best_skill <= 0:
            continue
        sv = r.get("skill_f")
        if sv is not None and sv > 0:
            r["skill_w"] = min(100, round(100 * sv / best_skill))
    for r in others:
        r["rank"] = None
        r["mark"] = ""
        r["mark_title"] = ""
        r["best_mae"] = r["best_skill"] = False
        # A1: the baseline's own n is its whole record, not the intersection the skill column
        # divides by, and saying so is the only way the two numbers stop contradicting each other.
        if r["baseline"]:
            r["n_note"] = "all days"
    return {"lead": lead, "rows": ranked + others, "leader": leader,
            "leader_name": _mname(model_idx, leader) if leader else "",
            "n_ranked": len(ranked), "any_ranked": bool(ranked),
            "holm": _has_holm(pw_sub)}


_MARK_TITLE = {
    "★": "lowest MAE among ranked models (n >= 30) in this view",
    "=": "not distinguishable from the leader after the Holm correction within this table",
    "▼": "worse than the leader; distinguishable after the Holm correction within this table",
    "▲": "better than the leader on their common days; distinguishable after the Holm correction",
    "·": "no paired comparison available",
}

#: Columns that may carry the corrected verdict, newest name first.
_HOLM_COLUMNS = ("distinguishable_holm",)
_UNCORRECTED_COLUMNS = ("distinguishable_uncorrected", "significant")


def _has_holm(pw_sub: pd.DataFrame | None) -> bool:
    return (pw_sub is not None and len(pw_sub) > 0
            and any(c in pw_sub.columns for c in _HOLM_COLUMNS))


def _flag(row, columns: tuple[str, ...]):
    for c in columns:
        v = row.get(c)
        if v is not None and not _isnan(v):
            return bool(v)
    return None


def _significance(pw_sub: pd.DataFrame, lead: int, leader: str | None) -> dict[str, str]:
    """Symbol per model: is its MAE distinguishable from the leader's *after* the Holm correction?

    Only ``distinguishable_holm`` is marked (review B2).  Every displayed table compares a family
    of models against one leader, so an uncorrected 95 % interval would call roughly one in twenty
    of them apart by chance; the uncorrected verdict is still published, but only in the pairwise
    table on the permanent link, where it is one labelled column among several.
    """
    if leader is None or pw_sub is None or len(pw_sub) == 0:
        return {}
    if not _has_holm(pw_sub):
        return {}
    g = pw_sub[pw_sub["lead_day"] == int(lead)]
    if g.empty:
        return {}
    out: dict[str, str] = {}
    for _, r in g.iterrows():
        if r["model_a"] == leader:
            other, sign = r["model_b"], -1.0
        elif r["model_b"] == leader:
            other, sign = r["model_a"], 1.0
        else:
            continue
        flag = _flag(r, _HOLM_COLUMNS)
        if not flag:
            out[other] = "="
        else:
            diff = sign * float(r["mae_diff"])  # other − leader
            out[other] = "▼" if diff > 0 else "▲"
    return out


def _lead_matrix(sub: pd.DataFrame, model_idx: dict, leads: list[int]) -> dict:
    """models × lead days, MAE over bias, every cell a link to its permanent page."""
    if sub.empty:
        return {}
    use_leads = [ld for ld in leads if ld in set(sub["lead_day"])]
    rows = []
    vmax = 0.0
    for mid in sorted(sub["model_id"].unique().tolist()):
        q = sub[sub["model_id"] == mid].set_index("lead_day")
        cells, spark = [], []
        for ld in use_leads:
            if ld not in q.index:
                cells.append(None)
                continue
            r = q.loc[ld]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            v = _row_view(r, ALL_STATIONS, mid, ld, model_idx)
            cells.append(v)
            if v["mae_f"]:
                vmax = max(vmax, v["mae_f"])
        for ld in SPARK_LEADS:
            if ld in q.index:
                r = q.loc[ld]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                spark.append(_f(r["mae"]))
            else:
                spark.append(None)
        rows.append({
            "model_id": mid, "name": _mname(model_idx, mid), "cells": cells, "spark": spark,
            "baseline": model_idx.get(mid, {}).get("baseline", False),
        })
    for row in rows:
        row["spark_svg"] = Markup(svg.sparkline(
            row["spark"], label=f"{row['name']} MAE by lead day 1 to 9", vmax=vmax or None))
    return {"leads": use_leads, "rows": rows, "spark_leads": list(SPARK_LEADS)}


def _bias_maps(sc: pd.DataFrame, station_idx: dict, view, model_idx: dict | None = None
               ) -> list[dict]:
    """Two maps: one fixed reference model, and the mean over all models (review A6).

    The previous map coloured each station by the bias of *its own best model*, which is a
    winner's-curse picture: on 16–28 days the winner at a station is largely noise, and the map
    invited exactly the reading ("this model is better here") that the sample cannot support.  A
    fixed reference and an all-model mean are both answerable questions.
    """
    window, init_hour, method, variable = view
    sub = sc[(sc["station_id"] != ALL_STATIONS) & (sc["window"] == window)
             & (sc["init_hour"] == int(init_hour)) & (sc["method"] == method)
             & (sc["variable"] == variable) & (sc["lead_day"] == 1)
             & (sc["model_id"] != PERSISTENCE_ID)]
    if sub.empty:
        return []
    label_var = VAR_LABEL.get(variable, variable)
    scope = f"lead day 1, {label_var}, {window} window, {int(init_hour):02d}Z, {method}"
    maps = []

    ref = sub[sub["model_id"] == REFERENCE_MODEL]
    if not ref.empty:
        points, rows = _map_rows(ref, station_idx, model_idx, per_model=True)
        maps.append({
            "key": "reference",
            "title": f"{_mname(model_idx, REFERENCE_MODEL)} bias",
            "caption": (f"Mean bias of the fixed reference model "
                        f"{_mname(model_idx, REFERENCE_MODEL)} ({scope}). One model everywhere, so "
                        f"the colours compare stations, not models."),
            "svg": Markup(svg.us_map(points, label=f"Mean bias in °F of "
                                                   f"{_mname(model_idx, REFERENCE_MODEL)}, {scope}")),
            "rows": rows, "per_model": True,
        })

    points, rows = _map_rows(sub, station_idx, model_idx, per_model=False)
    maps.append({
        "key": "mean",
        "title": "All-model mean bias",
        "caption": (f"Mean bias averaged over every scored model at each station ({scope}). A "
                    f"station that is cold for all of them is a station property — elevation, "
                    f"coastline, a grid cell that is partly sea — not a model ranking."),
        "svg": Markup(svg.us_map(points, label=f"All-model mean bias in °F, {scope}")),
        "rows": rows, "per_model": False,
    })
    return maps


def _map_rows(sub: pd.DataFrame, station_idx: dict, model_idx: dict | None,
              per_model: bool) -> tuple[list[dict], list[dict]]:
    points, rows = [], []
    for sid, grp in sub.groupby("station_id", observed=True):
        st = station_idx.get(sid)
        if per_model:
            r0 = grp.iloc[0]
            bias_c = float(r0["bias"]) if not _isnan(r0["bias"]) else None
            mae_s = f_delta(r0["mae"])
            n = int(r0["n"])
            sig = not (_isnan(r0.get("bias_ci_low")) or _isnan(r0.get("bias_ci_high"))) and not (
                float(r0["bias_ci_low"]) <= 0.0 <= float(r0["bias_ci_high"]))
            note = _mname(model_idx, r0["model_id"])
            n_models = 1
        else:
            bias_c = float(grp["bias"].mean()) if grp["bias"].notna().any() else None
            mae_s = f_delta(grp["mae"].mean())
            n = int(grp["n"].max())
            sig = True  # a mean over models has no interval of its own; colour by magnitude
            n_models = int(grp["model_id"].nunique())
            note = f"{n_models} models"
        bias_f = None if bias_c is None else bias_c * C_TO_F_DELTA
        row = {
            "id": sid, "name": st.name if st else sid,
            "lat": st.lat if st else None, "lon": st.lon if st else None,
            "n": n, "n_models": n_models, "note": note,
            "mae": mae_s, "bias": "—" if bias_c is None else f_signed(bias_c),
            "bias_class": svg.bias_class(bias_f, sig),
            "bias_short": _BIAS_SHORT.get(svg.bias_class(bias_f, sig), "z0"),
            "sign": "" if not bias_f else ("+" if bias_f > 0 else "−"),
            "href": f"/station/{sid}/",
            "low_n": n < MIN_N,
        }
        rows.append(row)
        if row["lat"] is not None:
            points.append(row)
    rows.sort(key=lambda r: r["id"])
    return points, rows


def _segment_note(sub: pd.DataFrame) -> dict:
    """The model version and segment start the published scores actually cover (METHODOLOGY §4).

    Scores are restricted to the newest model-version segment, so the version string and its start
    date belong next to every number the model produces.
    """
    out = {"model_version": "", "segment_start": ""}
    if sub is None or len(sub) == 0:
        return out
    if "model_version" in sub:
        vals = [v for v in sub["model_version"].dropna().unique().tolist() if str(v)]
        out["model_version"] = ", ".join(sorted(str(v) for v in vals))
    if "segment_start" in sub:
        starts = pd.to_datetime(sub["segment_start"], errors="coerce").dropna()
        if len(starts):
            out["segment_start"] = starts.max().date().isoformat()
    return out


def _availability(avail: pd.DataFrame, model_idx: dict | None = None):
    if avail is None or avail.empty:
        return [], "", ""
    avail = avail.sort_values("model_id").copy()
    avail["_s"] = pd.to_datetime(avail["period_start"])
    avail["_e"] = pd.to_datetime(avail["period_end"])
    lo, hi = avail["_s"].min(), avail["_e"].max()
    span = max((hi - lo).days, 1)
    out = []
    for _, r in avail.iterrows():
        s, e = r["_s"], r["_e"]
        left = 100.0 * (s - lo).days / span
        width = max(100.0 * ((e - s).days + 1) / span, 1.0)
        out.append({
            "model_id": r["model_id"], "name": _mname(model_idx, r["model_id"]), "n": int(r["n"]),
            "period": f_period(r["period_start"], r["period_end"]),
            "left": round(left, 2), "width": round(min(width, 100.0 - left), 2),
        })
    return out, lo.date().isoformat(), hi.date().isoformat()


def _grid_html(env, template, base_ctx, sub, view2, leads, model_idx, *, row_key,
               station_fixed=None, model_fixed=None, extra=None):
    window, init_hour = view2
    method = DEFAULT_METHOD
    part = sub[
        (sub["window"] == window)
        & (sub["init_hour"] == int(init_hour))
        & (sub["method"] == method)
    ]
    blocks = _var_blocks(part, PAGE_VARIABLES, leads, model_idx, row_key,
                         station_fixed, model_fixed)
    secondary = _var_blocks(part, CLI_VARIABLES, leads, model_idx, row_key,
                            station_fixed, model_fixed)
    ctx = dict(base_ctx)
    ctx.update(extra or {})
    return env.get_template(template).render(
        **ctx, blocks=blocks, secondary_blocks=secondary, leads=leads,
        spark_leads=list(SPARK_LEADS),
        window=window,
        window_label="all available history" if window == "all" else f"the last {window[:-1]} days",
        init_hour=f"{int(init_hour):02d}", method=method,
    )


def _sentence(text: str) -> str:
    """First letter upper-cased and nothing else touched, so ``NWS CLI`` survives a heading."""
    return text[:1].upper() + text[1:] if text else text


def _var_blocks(part, variables, leads, model_idx, row_key, station_fixed, model_fixed):
    """One MAE grid per variable that has any row, in the order given."""
    blocks = []
    for var in variables:
        p = part[part["variable"] == var]
        if p.empty:
            continue
        rows, vmax = [], 0.0
        for key in sorted(p[row_key].unique().tolist()):
            q = p[p[row_key] == key].set_index("lead_day")
            cells, spark = [], []
            for lead in leads:
                if lead in q.index:
                    r = q.loc[lead]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    sid = station_fixed if station_fixed is not None else key
                    mid = model_fixed if model_fixed is not None else key
                    bias_f = _f(r["bias"])
                    nf = r.get("n_flagged")
                    cells.append({
                        "mae": f_delta(r["mae"]), "bias": f_signed(r["bias"]),
                        "bias_class": svg.bias_class(bias_f),
                        "bias_short": _BIAS_SHORT.get(svg.bias_class(bias_f), "z0"),
                        "n": int(r["n"]),
                        "low_n": int(r["n"]) < MIN_N,
                        "n_flagged": 0 if _isnan(nf) else int(nf),
                        "permalink": permalink_url(sid, mid, lead),
                    })
                else:
                    cells.append(None)
            for lead in SPARK_LEADS:
                if lead in q.index:
                    r = q.loc[lead]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    v = _f(r["mae"])
                    spark.append(v)
                    if v:
                        vmax = max(vmax, v)
                else:
                    spark.append(None)
            rows.append({
                "key": key,
                "name": _mname(model_idx, key) if row_key == "model_id" else key,
                "cells": cells, "spark": spark,
                "baseline": bool(model_idx.get(key, {}).get("baseline", False))
                if row_key == "model_id" else False,
            })
        for row in rows:
            row["spark_svg"] = Markup(svg.sparkline(
                row["spark"], label=f"{row['name']} MAE by lead day 1 to 9, {var}",
                vmax=vmax or None))
        flagged = max((c["n_flagged"] for row in rows for c in row["cells"] if c), default=0)
        blocks.append({"variable": var, "label": _sentence(VAR_LABEL.get(var, var)),
                       "truth": VAR_TRUTH.get(var, "the matching observation"),
                       "headline": var == HEADLINE_VARIABLE,
                       "rows": rows, "flagged": flagged})
    return blocks


def _pair_views(grp: pd.DataFrame | None, model_id: str,
                model_idx: dict | None = None) -> list[dict]:
    if grp is None or len(grp) == 0:
        return []
    out = []
    for _, r in grp.iterrows():
        if r["model_a"] == model_id:
            other, sign = r["model_b"], 1.0
        elif r["model_b"] == model_id:
            other, sign = r["model_a"], -1.0
        else:
            continue
        lo, hi = r["ci_low"], r["ci_high"]
        if sign < 0:
            lo, hi = (None if _isnan(hi) else -hi), (None if _isnan(lo) else -lo)
        holm = _flag(r, _HOLM_COLUMNS)
        unc = _flag(r, _UNCORRECTED_COLUMNS)
        p_boot = r.get("p_boot")
        out.append({
            "other": other,
            "other_name": _mname(model_idx, other),
            "n_common": int(r["n_common"]),
            "diff": f_signed(sign * r["mae_diff"]),
            "ci": f_ci(lo, hi),
            "p_boot": "—" if _isnan(p_boot) else f"{float(p_boot):.3f}",
            "holm": "—" if holm is None else ("yes" if holm else "no"),
            "uncorrected": "—" if unc is None else ("yes" if unc else "no"),
        })
    return sorted(out, key=lambda d: d["other"])


def _recent_truth(truth_sel: pd.DataFrame, station_id: str) -> list[dict]:
    if truth_sel is None or len(truth_sel) == 0 or station_id == ALL_STATIONS:
        return []
    t = truth_sel[truth_sel["station_id"] == station_id]
    if t.empty:
        return []
    wide = t.pivot_table(index="climo_date", columns="variable", values="obs_c", aggfunc="first")
    meta = t.sort_values("climo_date").groupby("climo_date").agg(
        source=("truth_source", "first"), flag=("qc_flag", "first"))
    wide = wide.join(meta).sort_index(ascending=False).head(RECENT_TRUTH_DAYS)
    out = []
    for idx, r in wide.iterrows():
        out.append({
            "date": pd.Timestamp(idx).date().isoformat(),
            "tmax": "—" if _isnan(r.get("tmax")) else f"{float(r['tmax']) * 9 / 5 + 32:.0f}",
            "tmin": "—" if _isnan(r.get("tmin")) else f"{float(r['tmin']) * 9 / 5 + 32:.0f}",
            "source": r.get("source", ""),
            "flag": r.get("flag", "") or "",
        })
    return out


# ------------------------------------------------------------------------------------------
# permanent links
# ------------------------------------------------------------------------------------------

#: Display order of the variables on a permanent link: headline, per-hour, like-for-like, then
#: the secondary CLI comparisons.
_VARIABLE_ORDER = {v: i for i, v in enumerate(
    (HEADLINE_VARIABLE, *HOUR_VARIABLES, *SAMPLED_VARIABLES, *CLI_VARIABLES))}


#: Everything a reader compares between two rows of the permanent-link table.  Two rows that are
#: identical on all of it are the same numbers computed over the same days, printed twice.
_ROW_SIGNATURE = ("n", "n_flagged", "mae", "mae_ci", "mae_debiased", "bias", "bias_ci", "rmse",
                  "rmse_ci", "hit1f", "hit1f_ci", "hit2f", "hit3f", "skill", "skill_ci",
                  "skill_vs", "skill_debiased", "n_debiased", "period", "low_n")


def _merge_equal_windows(rows: list[dict]) -> list[dict]:
    """Print one row where two windows cover exactly the same days.

    While the record is shorter than a year, ``365d`` and ``all`` are the *same set of days*, so
    the table repeats every number for them; the same happens to ``90d`` in the first three
    months.  Rows arrive sorted by (variable, initialization, interpolation, window), so identical
    neighbours are adjacent: they are collapsed into one row whose key names both windows.  The
    moment the record outgrows a window the rows differ again and the merge stops happening, with
    no threshold to maintain.
    """
    out: list[dict] = []
    for r in rows:
        prev = out[-1] if out else None
        if (prev is not None
                and prev["variable"] == r["variable"]
                and prev["init_hour"] == r["init_hour"]
                and prev["method"] == r["method"]
                and all(prev[k] == r[k] for k in _ROW_SIGNATURE)):
            prev["windows"].append(r["window"])
            prev["window_label"] = " · ".join(prev["windows"])
            continue
        row = dict(r)
        row["windows"] = [r["window"]]
        row["window_label"] = r["window"]
        out.append(row)
    return out


def _write_permalinks(env, w, out, base_ctx, sc, pw, model_idx, station_idx, series_idx,
                      month_idx, allday_idx, as_of_s, data_through) -> int:
    chart_variable = base_ctx.get("chart_variable", DEFAULT_VARIABLE)
    pw_head = pd.DataFrame()
    if len(pw):
        pw_head = pw[
            (pw["window"] == DEFAULT_WINDOW)
            & (pw["init_hour"] == DEFAULT_INIT)
            & (pw["method"] == DEFAULT_METHOD)
            & (pw["variable"] == chart_variable)
        ]
    pw_idx: dict[tuple[str, int], pd.DataFrame] = {}
    if len(pw_head):
        for key, grp in pw_head.groupby(["station_id", "lead_day"], observed=True):
            pw_idx[(key[0], int(key[1]))] = grp

    tpl = env.get_template("permalink.html")
    n = 0
    for (sid, mid, lead), grp in sc.groupby(["station_id", "model_id", "lead_day"], observed=True):
        lead = int(lead)
        grp = grp.assign(_w=grp["window"].map(WINDOW_ORDER).fillna(9),
                         _v=grp["variable"].map(_VARIABLE_ORDER).fillna(99))
        all_rows = [_row_view(r, sid, mid, lead, model_idx) for _, r in
                    grp.sort_values(["_v", "variable", "init_hour", "method", "_w"]).iterrows()]
        rows = _merge_equal_windows([r for r in all_rows if not r["secondary"]])
        secondary_rows = _merge_equal_windows([r for r in all_rows if r["secondary"]])
        pairs = _pair_views(pw_idx.get((sid, lead)), mid, model_idx)
        st = station_idx.get(sid)
        st_name = st.name if st else ("All stations (mean of daily station errors)"
                                      if sid == ALL_STATIONS else sid)
        key = (sid, mid, lead)
        ser = series_idx.get(key)
        chart = Markup(svg.line_chart(
            ser["dates"] if ser else [], ser["values"] if ser else [],
            label=(f"Daily forecast minus observed error in °F for {_mname(model_idx, mid)} at "
                   f"{sid}, lead day {lead}, {DEFAULT_INIT:02d}Z {DEFAULT_METHOD} "
                   f"{chart_variable}")))
        hist_svg, hist_rows = svg.histogram(
            allday_idx.get(key, []),
            label=(f"Distribution of the signed daily error in °F for {_mname(model_idx, mid)} "
                   f"at {sid}, lead day {lead}"))
        series_rows = []
        if ser:
            series_rows = [
                {"date": d, "err": "—" if v is None else _minus(f"{v:+.2f}")}
                for d, v in zip(ser["dates"], ser["values"])
            ][::-1]
        base = f"station/{sid}/model/{mid}/lead/{lead}"
        _write_permalink_csv(out / base / "errors.csv", sid, mid, lead, ser, chart_variable)
        html = tpl.render(
            **base_ctx,
            station_id=sid, model_id=mid, lead=lead,
            station_name=st_name,
            station=st,
            model_name=_mname(model_idx, mid),
            model_family=model_idx.get(mid, {}).get("family", mid),
            model_info=model_idx.get(mid, {}),
            segment=_segment_note(grp),
            n_flagged=max((r["n_flagged"] for r in all_rows), default=0),
            rows=rows, secondary_rows=secondary_rows,
            pairwise=pairs, pairwise_window=DEFAULT_WINDOW,
            holm=any(p["holm"] != "—" for p in pairs),
            chart_init=f"{DEFAULT_INIT:02d}", chart_method=DEFAULT_METHOD,
            chart_window=DEFAULT_WINDOW,
            series_days=SERIES_DAYS,
            series_svg=chart, series_rows=series_rows,
            hist_svg=Markup(hist_svg), hist_rows=hist_rows,
            months=month_idx.get(key, [])[::-1][:24],
            availability=_availability(grp[(grp["window"] == "all")
                                           & (grp["variable"] == chart_variable)],
                                       model_idx)[0],
            # The per-card JSON is a fragment of the station's bundle: one file instead of the
            # 1 800 that each repeated the envelope and the station's whole scores table.
            json_url=f"/station/{sid}/cards.json#{mid}-{lead}",
            csv_url=f"/{base}/errors.csv",
            citation=citation(sid, mid, lead, as_of_s),
            citation_long=citation_long(sid, st_name, mid, lead, data_through, as_of_s),
            canonical=f"{SITE_URL}{permalink_url(sid, mid, lead)}",
        )
        w.write(f"{base}/index.html", html)
        n += 1
    return n


def _write_permalink_csv(path: Path, sid: str, mid: str, lead: int, ser: dict | None,
                         variable: str = DEFAULT_VARIABLE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["station_id,model_id,lead_day,init_hour,method,variable,climo_date,error_f"]
    if ser:
        for d, v in zip(ser["dates"], ser["values"]):
            if v is None:
                continue
            lines.append(f"{sid},{mid},{lead},{DEFAULT_INIT},{DEFAULT_METHOD},"
                         f"{variable},{d},{v:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------
# fixed pages, downloads, feed and headers
# ------------------------------------------------------------------------------------------

def _prune_variants(out: Path) -> None:
    """Delete the generated ``v/`` view directories so a renamed slug cannot survive a rebuild.

    Also removes the ``name 2`` copies macOS creates when a directory is written twice while a
    sync client holds it open: they are duplicates of real pages, they get deployed, and they
    count against the 20 000-file limit of Cloudflare Pages.
    """
    for path in (out / "v", *(out / "station").glob("*/v"), *(out / "model").glob("*/v")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for path in out.rglob("* 2"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            log.info("removed macOS duplicate directory %s", path)
    for path in list(out.rglob("* 2.*")):
        if path.is_file():
            path.unlink(missing_ok=True)


def _tree_size(out: Path) -> tuple[int, int]:
    """``(files, bytes)`` of everything that will be deployed."""
    n = total = 0
    for p in out.rglob("*"):
        if p.is_file():
            n += 1
            total += p.stat().st_size
    return n, total


def _write_assets(out: Path) -> dict[str, str]:
    """Copy the assets under a content-hashed name and return the URLs to link.

    The plain ``/assets/site.css`` is cached for an hour by ``_headers``, so a deploy that changes
    the stylesheet would otherwise leave every returning visitor on the previous one for up to an
    hour — a real bug, not a theoretical one. Pages therefore link ``site.<hash>.css``, which
    changes whenever the bytes change. The unhashed name is still written so that an external
    link to it keeps working.
    """
    asset_dir = out / "assets"
    if asset_dir.is_dir():
        shutil.rmtree(asset_dir, ignore_errors=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    urls: dict[str, str] = {}
    for f in sorted(ASSETS.glob("*")):
        # Only what the pages link. The assets directory also holds generated Python data (the
        # public-domain coastline path), which belongs in the build, not in public/.
        if not f.is_file() or f.suffix not in {".css", ".js", ".svg", ".woff2"}:
            continue
        data = f.read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:10]
        hashed = f"{f.stem}.{digest}{f.suffix}"
        (asset_dir / hashed).write_bytes(data)
        (asset_dir / f.name).write_bytes(data)
        urls[f"asset_{f.stem}"] = f"/assets/{hashed}"
    return urls


def _write_headers(out: Path) -> Path:
    """Cloudflare Pages ``_headers``: CORS and caching for the API, per docs/05 §D."""
    body = """# Generated by castcheck.site.build — do not edit by hand.
/api/*
  Access-Control-Allow-Origin: *
  Access-Control-Allow-Methods: GET, HEAD, OPTIONS
  Cache-Control: public, max-age=3600, stale-while-revalidate=86400

/data/*.csv
  Access-Control-Allow-Origin: *
  Cache-Control: public, max-age=3600, stale-while-revalidate=86400

/data/*.csv.gz
  Access-Control-Allow-Origin: *
  Cache-Control: public, max-age=3600, stale-while-revalidate=86400

/feed.xml
  Cache-Control: public, max-age=3600, stale-while-revalidate=86400

/assets/*
  Cache-Control: public, max-age=3600, stale-while-revalidate=86400

/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
"""
    path = out / "_headers"
    path.write_text(body, encoding="utf-8")
    return path


def _write_methodology(env, w, base_ctx) -> None:
    path = REPO_ROOT / "METHODOLOGY.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Methodology\n\nNot available."
    w.write("methodology/index.html", env.get_template("page.html").render(
        **base_ctx, heading="Methodology", body=_render_markdown(text)))


def _write_404(env, w, base_ctx) -> None:
    """Cloudflare Pages serves /404.html for unknown paths; without it a stale URL returns the home page with 200."""
    body = (
        "<p>There is nothing at this address. Permanent links have the form "
        "<code>/station/{ICAO}/model/{model_id}/lead/{day}/</code>; the "
        "<a href=\"/stations/\">stations</a> and <a href=\"/models/\">models</a> indexes list every valid combination, "
        "and the <a href=\"/data/\">data page</a> lists everything that can be downloaded.</p>"
    )
    w.write("404.html", env.get_template("page.html").render(**base_ctx, heading="Page not found", body=body))


def _write_status(env, w, base_ctx, report, names: dict[str, str] | None = None) -> None:
    view = _status_view(report, names)
    w.write("status/index.html", env.get_template("status.html").render(
        **base_ctx, report=report, **view))


def _status_view(report: dict | None, names: dict[str, str] | None = None) -> dict:
    """Turn the raw completeness report into uptime bars and a headline state."""
    if not report:
        return {"model_bars": [], "truth_bars": [], "instant_bars": [], "overall": "unknown",
                "last_run_human": "—", "n_pending": 0,
                "overall_text": "No status report has been generated yet."}
    days = report.get("days", 0)

    def bar(rows, key_yes, key_part, label):
        # A day the model was never expected to have — before it entered the record, an
        # initialization the upstream archive did not produce, or one whose publication deadline
        # has not passed yet — is drawn as an empty slot and left out of the percentage entirely
        # (review B8; `not_due_yet` from castcheck.schedule).
        flags = []
        for d in rows:
            if d.get("reason") == "not_due_yet":
                flags.append("wait")
            elif d.get("expected", True) is False:
                flags.append("na")
            elif d.get(key_yes):
                flags.append("yes")
            else:
                flags.append("part" if d.get(key_part) else "no")
        counted = [f for f in flags if f not in ("na", "wait")]
        pct = 100.0 * sum(1 for f in counted if f == "yes") / max(len(counted), 1)
        return {"svg": Markup(svg.availability_row(flags, label=label)),
                "uptime": f"{pct:.1f}%" if counted else "—", "flags": flags,
                "n_counted": len(counted),
                "n_wait": sum(1 for f in flags if f == "wait"),
                "n_na": sum(1 for f in flags if f == "na")}

    last_run_human = human_time(report.get("generated_at"))
    names = names or display_names()
    model_bars = []
    for m in report.get("models", []):
        label = names.get(m["model_id"], m["model_id"])
        b = bar(m.get("days", []), "complete", "stations_any",
                f"{label} {m['init_hour']:02d}Z completeness, last {days} days")
        model_bars.append({**m, **b, "display": label})
    truth_bars = []
    for t in report.get("truth", []):
        b = bar(t.get("days", []), "cli_final", "any_source",
                f"{t['station_id']} truth availability, last {days} days")
        truth_bars.append({**t, **b})
    instant_bars = []
    for t in report.get("truth_instant", []):
        b = bar(t.get("days", []), "complete", "any_source",
                f"{t['station_id']} observed-instant coverage, last {days} days")
        instant_bars.append({**t, **b})

    # "Missing" means *due and absent*.  A 12Z run at 13 UTC, or a Los Angeles climate report at
    # 06 UTC, has not been published yet by anyone and is not a fault of this pipeline; those are
    # counted as pending, and the bar stays green while it says so (castcheck.schedule).
    n_pending = int(report.get("n_pending", 0) or 0)
    if report.get("ok") and n_pending:
        overall = "ok"
        text = (f"All due runs present — nothing that should exist yet is missing. "
                f"{n_pending} item(s) for the current day are not due yet.")
    elif report.get("ok"):
        overall, text = "ok", "All systems operational — nothing is missing for the current day."
    elif report.get("n_current_gaps", 0) > 0:
        overall = "bad"
        text = (f"{report['n_current_gaps']} due item(s) missing for the current day; "
                f"{report.get('n_gaps', 0)} over the last {days} days"
                + (f"; {n_pending} more are not due yet." if n_pending else "."))
    else:  # pragma: no cover - defensive
        overall, text = "warn", "Degraded."

    # A per-model state badge, so a reader scanning the column sees the shape before the numbers.
    for b in model_bars:
        pct = _pct_of(b["uptime"])
        b["state"] = "ok" if pct is not None and pct >= 99.0 else (
            "warn" if pct is not None and pct >= 90.0 else "bad")
        b["state_text"] = {"ok": "healthy", "warn": "degraded", "bad": "gaps"}[b["state"]]

    # The month bands over the day-by-day grid.
    month_groups = []
    for d in report.get("dates", []):
        label = _MONTH_LABEL(d)
        if month_groups and month_groups[-1]["label"] == label:
            month_groups[-1]["span"] += 1
        else:
            month_groups.append({"label": label, "span": 1})

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return None if not vals else sum(vals) / len(vals)

    kpi_runs = _mean([_pct_of(b["uptime"]) for b in model_bars])
    kpi_truth = _mean([_pct_of(b["uptime"]) for b in truth_bars])
    kpi_instants = _mean([_pct_of(b["uptime"]) for b in instant_bars])
    latest = sorted({m.get("latest_init") for m in model_bars if m.get("latest_init")})
    return {"model_bars": model_bars, "truth_bars": truth_bars, "instant_bars": instant_bars,
            "overall": overall, "overall_text": text, "month_groups": month_groups,
            "kpi_runs": "—" if kpi_runs is None else f"{kpi_runs:.1f}",
            "kpi_truth": "—" if kpi_truth is None else f"{kpi_truth:.1f}",
            "kpi_instants": "—" if kpi_instants is None else f"{kpi_instants:.1f}",
            "n_pending": n_pending,
            "last_run_human": last_run_human,
            "latest_init": latest[-1] if latest else "—"}


def _pct_of(text: str) -> float | None:
    """``"92.6%"`` back to ``92.6`` — the bars print the string, the KPIs average the number."""
    try:
        return float(str(text).rstrip("%"))
    except ValueError:
        return None


def _MONTH_LABEL(iso: str) -> str:  # noqa: N802 - a formatting helper, not a constant
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%B %Y")
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(iso)[:7]


#: 6.5 K per km is the standard-atmosphere lapse rate; the first-order magnitude of the
#: representativeness error a station's height difference imposes (review B7, DESIGN §10.4).
LAPSE_K_PER_M = 6.5 / 1000.0


def _station_field(st, name: str):
    """A station attribute that may not exist yet in this build's ``config.Station``."""
    return getattr(st, name, None)


def _station_dz(st) -> float | None:
    dz = _station_field(st, "dz_m")
    if dz is not None:
        return round(float(dz), 1)
    elev, grid = _station_field(st, "elev_m"), _station_field(st, "grid_elev_m")
    if elev is None or grid is None:
        return None
    return round(float(elev) - float(grid), 1)


def _station_lapse(st) -> float | None:
    dz = _station_dz(st)
    return None if dz is None else round(abs(dz) * LAPSE_K_PER_M, 2)


def _write_downloads(out: Path, scores, pairwise, errors, stations, models) -> list[dict]:
    """Write the bulk CSVs under ``/data/`` and describe them for the page."""
    d = out / "data"
    d.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    def add(name: str, what: str, n_rows: int) -> None:
        p = d / name
        if not p.exists():
            return
        items.append({"name": name, "href": f"/data/{name}", "what": what,
                      "size": _human_size(p.stat().st_size), "rows": n_rows})

    # gzip: the uncompressed table passed 20 MB in August 2026 and Cloudflare Pages refuses a
    # single file above 25 MiB, so the plain .csv is not published at all.
    (d / "scores_latest.csv").unlink(missing_ok=True)
    with gzip.open(d / "scores_latest.csv.gz", "wt", encoding="utf-8", newline="") as f:
        if scores is not None and len(scores):
            # 3 dp in °C is 0.001 °C — the same precision the JSON API publishes, two orders of
            # magnitude finer than the whole-°F truth, and it keeps the published CSV a third
            # smaller than float64 repr would.
            scores.round(3).to_csv(f, index=False)
        else:
            f.write(",".join(SCORE_COLUMNS) + "\n")
    add("scores_latest.csv.gz", "every published aggregate: station × model × init × lead × "
        "variable × method × window, with n, MAE, bias, RMSE, hit rates, skill and bootstrap "
        "intervals (values in °C).", len(scores) if scores is not None else 0)

    # The full pairwise table is 6 MB compressed and grows as the square of the model count; it
    # is a bulk archive, not something a page needs, so it is published on Hugging Face and the
    # slice the site actually shows (station_id=ALL) stays here as JSON at
    # /api/v1/pairwise/latest.json.  Delete a copy left by an earlier build.
    (d / "pairwise_latest.csv.gz").unlink(missing_ok=True)
    (d / "pairwise_latest.csv").unlink(missing_ok=True)
    items.append({
        "name": "pairwise_latest.csv.gz", "href": f"{HF_FILES_URL}/data/scores", "external": True,
        "what": "paired model-vs-model MAE differences on common days, with the bootstrap "
                "interval, the bootstrap p-value and the Holm verdict (°C). Published on Hugging "
                "Face: it grows as the square of the model count and nothing on this site reads "
                "it. The station_id=ALL slice the leaderboards use is at "
                "/api/v1/pairwise/latest.json.",
        "size": "on Hugging Face",
        "rows": len(pairwise) if pairwise is not None else 0})

    # One file per station, not one file for everything: the combined table reached 25 MB — the
    # size at which Cloudflare Pages refuses a file outright — as soon as the pooled t2 variable
    # added four rows per day. Sharding by station is the same cut the JSON API uses, keeps each
    # file a couple of MB, and is what someone asking for one station actually wants.
    err_dir = d / "daily_errors"
    err_dir.mkdir(parents=True, exist_ok=True)
    for stale in err_dir.glob("*.csv.gz"):
        stale.unlink()
    (d / "daily_errors.csv.gz").unlink(missing_ok=True)
    cols = ["station_id", "model_id", "init_hour", "lead_day", "method", "variable",
            "sub", "climo_date", "fcst_c", "obs_c", "err"]
    shards: list[tuple[str, int]] = []
    if errors is not None and len(errors):
        e = errors[[c for c in cols if c in errors.columns]].copy()
        # `sub` carries the valid hour of the four rows a pooled t2 day has; without it those
        # four rows are indistinguishable in the download.
        e = e.rename(columns={"sub": "valid_hour_utc"})
        e["climo_date"] = pd.to_datetime(e["climo_date"]).dt.date
        for sid, grp in e.groupby("station_id", observed=True):
            with gzip.open(err_dir / f"{sid}.csv.gz", "wt", encoding="utf-8", newline="") as f:
                # 2 dp in °C is 0.01 °C — a fiftieth of the whole-°F step the observations are
                # reported in, and a quarter smaller on disk than the 4 dp this used to carry.
                grp.round(2).to_csv(f, index=False)
            shards.append((str(sid), len(grp)))
    for sid, n in shards:
        add(f"daily_errors/{sid}.csv.gz",
            f"every scored value at {sid}: one row per model, initialization, lead day, method, "
            f"variable and climatological day (four rows a day for the pooled t2), with the "
            f"forecast, the observation and the signed error (°C, 2 dp). Download all "
            f"{len(shards)} to recompute anything.", n)

    st_lines = ["station_id,name,cli_pil,iem_id,tz,std_offset_h,lat,lon,elev_m,grid_elev_m,dz_m,"
                "lapse_k,market_city"]
    for s in stations:
        st_lines.append(",".join(str(x) if x is not None else "" for x in (
            s.id, f'"{s.name}"', s.cli_pil, _station_field(s, "iem_id") or "", s.tz,
            s.std_offset_h, s.lat, s.lon, s.elev_m,
            _station_field(s, "grid_elev_m"), _station_dz(s), _station_lapse(s),
            _station_field(s, "market_city") or "")))
    (d / "stations.csv").write_text("\n".join(st_lines) + "\n", encoding="utf-8")
    add("stations.csv", "station metadata: identifier, name, truth product, IEM archive id, fixed "
        "standard UTC offset, coordinates, station elevation, the mean elevation of the 0.25° grid "
        "cell, their difference and its first-order lapse-rate magnitude, and the "
        "temperature-contract city the station was selected for.", len(stations))

    md_lines = ["model_id,family,source,product,init_field,inits,step_h,max_h,native_extremes"]
    for m in models:
        md_lines.append(",".join((
            m.model_id, f'"{m.family}"', m.source, m.product, m.init_field or "",
            f'"{" ".join(str(i) for i in m.inits)}"', str(m.step_h), str(m.max_h),
            f'"{" ".join(m.native_extremes)}"')))
    md_lines.append("persistence,\"Persistence (baseline)\",truth,obs,,\"0 12\",24,240,\"\"")
    (d / "models.csv").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    add("models.csv", "model registry, including the persistence baseline.", len(models) + 1)
    return items


def _revision_impact(truth: pd.DataFrame | None) -> dict:
    """How much the first-final policy actually costs (review C4).

    Every score is computed from the *first* final CLI report and never rewritten; a later
    corrected report is stored alongside it.  This counts the corrections and measures how far
    they moved the number, which is the upper bound on what the policy hides.
    """
    empty = {"n_days": 0, "n_revised": 0, "rows": [], "max": None, "any": False}
    if truth is None or len(truth) == 0 or "revised" not in truth.columns:
        return empty
    t = truth.copy()
    t = t[t["source"] == "CLI"] if "source" in t.columns else t
    if "is_final" in t.columns:
        t = t[t["is_final"].fillna(False).astype(bool)]
    if not len(t):
        return empty
    rows = []
    biggest = 0.0
    n_revised = int(t["revised"].fillna(False).astype(bool).sum())
    for var, first, later in (("daily maximum", "tmax_f", "revised_tmax_f"),
                              ("daily minimum", "tmin_f", "revised_tmin_f")):
        if first not in t.columns or later not in t.columns:
            continue
        d = pd.to_numeric(t[later], errors="coerce") - pd.to_numeric(t[first], errors="coerce")
        d = d[d.notna() & (d != 0)]
        if not len(d):
            rows.append({"variable": var, "n": 0, "mean": "—", "p50": "—", "max": "—",
                         "share": "0%"})
            continue
        biggest = max(biggest, float(d.abs().max()))
        rows.append({
            "variable": var, "n": int(len(d)),
            "mean": f"{float(d.mean()):+.2f}",
            "p50": f"{float(d.median()):+.1f}",
            "max": f"{float(d.abs().max()):.0f}",
            "share": f"{100.0 * len(d) / max(len(t), 1):.2f}%",
        })
    return {"n_days": int(len(t)), "n_revised": n_revised, "rows": rows,
            "max": f"{biggest:.0f}" if biggest else None,
            "any": any(r["n"] for r in rows)}


def _write_data(env, w, base_ctx, api_written, scores, pairwise, downloads,
                truth: pd.DataFrame | None = None) -> None:
    api_written = api_written or {}
    endpoints = [
        ("scores/index.json", "the index of the sharded scores export: which stations, models, "
         "leads and views exist and where each shard is",
         api_written.get("scores/index.json", 1)),
        ("/station/{station}/cards.json", "the shard the index points at: every score for one "
         "station, plus one card per model and lead day (its pairwise table and daily error "
         "series). /api/v1/scores/by-station/{station}.json points here",
         api_written.get("station/{station}/cards.json", "—")),
        ("scores/leaderboard.json", "the station_id=ALL slice used by the front page",
         api_written.get("scores/leaderboard.json", "—")),
        ("leaderboard/{window}-{init}z-{method}-{variable}.json",
         f"one pre-built file per leaderboard view ({len(VIEWS)} of them)",
         api_written.get("leaderboard/{view}.json", "—")),
        ("/station/{station}/cards.json#{model}-{lead}",
         "one permanent-link card inside that bundle: its pairwise table and the last 90 days of "
         "daily errors",
         api_written.get("cards", "—")),
        ("pairwise/latest.json", "model-vs-model paired bootstrap (station_id=ALL)",
         api_written.get("pairwise/latest.json", len(pairwise))),
        ("stations.json", "station metadata", api_written.get("stations.json", "—")),
        ("models.json", "model metadata", api_written.get("models.json", "—")),
        ("status.json", "pipeline completeness report", api_written.get("status.json", "—")),
        ("diagnostics.json", "the diurnal-structure diagnostic behind /diagnostics/: bias at each "
         "of the four synoptic instants, bias against lead day, and the sampling penalty",
         api_written.get("diagnostics.json", "—")),
        ("openapi.json", "OpenAPI 3.1 description of the endpoints above",
         api_written.get("openapi.json", "—")),
    ]
    schemas = [
        {"name": "forecast_values", "what": "one extracted station value per model run, valid "
         "time, variable and interpolation method (DESIGN §3.1). This is the 6-hourly "
         "instantaneous layer: the only table that is untouched by any daily-extreme definition, "
         "and the one to start from to re-derive everything else or to check a claim about "
         "diurnal amplitude.",
         "columns": ["model_id", "model_version", "init_time", "valid_time", "lead_h",
                     "station_id", "variable", "bucket_h", "method", "value_c", "missing_reason",
                     "source_url", "fetched_at"]},
        {"name": "truth_instant", "what": "the observed 2 m temperature at the four common "
         "synoptic instants — the routine METAR nearest the hour within ±35 minutes (DESIGN "
         "§10.1). This is the truth for the headline t2 variables and for tmax_s/tmin_s.",
         "columns": _TRUTH_INSTANT_COLUMNS},
        {"name": "truth_daily", "what": "one row per station-day-source with the first-final "
         "policy and QC flags (DESIGN §3.2). Truth for the secondary tmax_cli/tmin_cli.",
         "columns": TRUTH_COLUMNS},
        {"name": "daily_forecasts", "what": "sampled and native daily extremes per model run, "
         "station and climatological day (DESIGN §3.3).", "columns": DAILY_COLUMNS},
        {"name": "scores", "what": "published aggregates with bootstrap intervals (DESIGN §3.4).",
         "columns": SCORE_COLUMNS},
        {"name": "pairwise", "what": "paired model-vs-model MAE differences (DESIGN §3.5).",
         "columns": PAIRWISE_COLUMNS},
    ]
    for t in schemas:
        per_table = COLUMN_DOCS_BY_TABLE.get(t["name"], {})
        t["fields"] = [
            {"name": c, "type": doc[0], "unit": doc[1], "what": doc[2]}
            for c, doc in (
                (c, per_table.get(c) or COLUMN_DOCS.get(c, ("—", "—", "")))
                for c in t["columns"]
            )
        ]
    w.write("data/index.html", env.get_template("data.html").render(
        **base_ctx,
        endpoints=[{"path": p, "href": f"/api/v1/{p}" if "{" not in p else "/api/v1/",
                    "what": what, "n": n} for p, what, n in endpoints],
        schemas=schemas,
        downloads=downloads,
        changelog=changelog_entries(),
        revisions=_revision_impact(truth),
        limitations=LIMITATIONS,
        data_citation=(
            f"CastCheck (2026). Station-level verification of raw weather-model 2 m temperature "
            f"forecasts [data set]. Methodology version {METHODOLOGY_VERSION}, schema version "
            f"{SCHEMA_VERSION}, data through {base_ctx['data_through']}. {SITE_URL} "
            f"doi:{CONCEPT_DOI}"),
    ))


# ------------------------------------------------------------------------------------------
# /diagnostics/ — the diurnal-structure diagnostic (METHODOLOGY §10.2)
# ------------------------------------------------------------------------------------------

#: Read as: at these UTC instants, this is the local standard time across the station set.  The
#: page prints the realised range from the registry rather than these words alone.
HOUR_WHEN = {
    0: "evening", 6: "night", 12: "early morning", 18: "midday to afternoon",
}

DIAGNOSTICS_LEDE = (
    "Under the four common synoptic instants the models do not carry the same bias at every hour "
    "of the day. This page publishes that structure and nothing else: it is an observation about "
    "these measurements, not yet attributed to a cause."
)

#: The three candidates METHODOLOGY §10.2 lists, in the same order and with the same wording, so
#: the page and the document cannot drift apart.  None is excluded by the data on this page.
DIAGNOSTICS_CANDIDATES = [
    ("Model diurnal amplitude",
     "the models may genuinely produce a flatter or sharper daily temperature cycle, in which "
     "case the hour-by-hour bias is a property of the forecast system"),
    ("The extreme-sampling penalty",
     "four samples a day cannot see the true daily maximum or minimum, and the size of that "
     "penalty depends on the shape of the curve being sampled — the third table below measures "
     "it directly (METHODOLOGY §2.3)"),
    ("The initial conditions",
     "the same architecture run from GFS and from IFS analyses differs by a large fraction of "
     "the effect, which points at near-surface temperature in the analysis rather than at the "
     "forecast model"),
]


def _series_styles(models: list[ModelSpec], model_idx: dict) -> dict[str, dict]:
    """``model_id -> {cls, alt, baseline}``: one hue per family, dashed for the second variant.

    Two initial-condition variants of one architecture are the same model, so they get the same
    colour and are separated by the dash pattern; every figure also prints the name beside the
    mark, so neither encoding is load-bearing on its own.
    """
    order: list[str] = []
    for m in models:
        if m.family not in order:
            order.append(m.family)
    seen: dict[str, int] = {}
    styles: dict[str, dict] = {}
    for m in models:
        rank = seen.get(m.family, 0)
        seen[m.family] = rank + 1
        styles[m.model_id] = {
            "cls": svg.SERIES_CLASSES[order.index(m.family) % len(svg.SERIES_CLASSES)],
            "alt": rank > 0,
            "baseline": False,
        }
    for mid, info in model_idx.items():
        if info.get("baseline"):
            styles[mid] = {"cls": svg.BASELINE_CLASS, "alt": False, "baseline": True}
    return styles


def _local_hour_span(stations: list[Station], hour_utc: int) -> str:
    """``"10:00-13:00"``: the local standard time of a UTC instant across the station set."""
    offs = sorted({int(s.std_offset_h) for s in stations if s.std_offset_h is not None})
    if not offs:
        return "—"
    lo, hi = (hour_utc + offs[0]) % 24, (hour_utc + offs[-1]) % 24
    return f"{lo:02d}:00" if lo == hi else f"{lo:02d}:00\u2013{hi:02d}:00"


def _diagnostics_view(payload: dict, models: list[ModelSpec], model_idx: dict,
                      stations: list[Station]) -> dict:
    """Everything ``diagnostics.html`` renders, in °F, from the same payload the API publishes."""
    names = _display_map(model_idx)
    styles = _series_styles(models, model_idx)
    hours = payload["hourly_bias"]["hours_utc"]
    cats = [f"{h:02d}Z" for h in hours]
    leads = payload["bias_by_lead"]["lead_days"]

    def style(mid: str) -> dict:
        return styles.get(mid, {"cls": svg.SERIES_CLASSES[0], "alt": False, "baseline": False})

    hour_groups, hour_rows = [], []
    for r in payload["hourly_bias"]["rows"]:
        mid = r["model_id"]
        st = style(mid)
        vals = [None if b is None else float(b) * C_TO_F_DELTA for b in r["bias"]]
        finite = [v for v in vals if v is not None]
        hour_groups.append({
            "name": names.get(mid, mid), "values": vals, "cls": st["cls"],
            "baseline": st["baseline"],
            "titles": [None if v is None else
                       f"{names.get(mid, mid)} {cats[i]}: {v:+.2f} °F, n = {f_int(r['n'][i])}"
                       for i, v in enumerate(vals)],
        })
        hour_rows.append({
            "id": mid, "name": names.get(mid, mid), "baseline": st["baseline"],
            "cls": st["cls"], "alt": st["alt"],
            "cells": [{"bias": f_signed(b), "n": f_int(n)}
                      for b, n in zip(r["bias"], r["n"], strict=True)],
            "range": "—" if len(finite) < 2 else _minus(f"{max(finite) - min(finite):.2f}"),
        })

    lead_series, lead_rows = [], []
    for r in payload["bias_by_lead"]["rows"]:
        mid = r["model_id"]
        st = style(mid)
        vals = [None if b is None else float(b) * C_TO_F_DELTA for b in r["bias"]]
        lead_series.append({"name": names.get(mid, mid), "values": vals, **st})
        lead_rows.append({
            "id": mid, "name": names.get(mid, mid), "baseline": st["baseline"],
            "cls": st["cls"], "alt": st["alt"],
            "cells": [{"bias": f_signed(b), "n": f_int(n)}
                      for b, n in zip(r["bias"], r["n"], strict=True)],
        })

    penalty_rows = []
    for r in payload["sampling_penalty"]["rows"]:
        mid = r["model_id"]
        st = style(mid)
        cells = []
        for sampled, cli in DIAGNOSTICS_PENALTY_PAIRS:
            pen = r.get(f"{sampled}_penalty")
            cells.append({
                "sampled": f_signed(r.get(f"{sampled}_bias")),
                "cli": f_signed(r.get(f"{cli}_bias")),
                "penalty": f_signed(pen),
                "penalty_f": None if pen is None else float(pen) * C_TO_F_DELTA,
                "n_sampled": f_int(r.get(f"{sampled}_n")),
                "n_cli": f_int(r.get(f"{cli}_n")),
            })
        penalty_rows.append({"id": mid, "name": names.get(mid, mid),
                             "baseline": st["baseline"], "cells": cells})
    ranked = [r for r in penalty_rows
              if not r["baseline"] and r["cells"][0]["penalty_f"] is not None]
    ranked.sort(key=lambda r: r["cells"][0]["penalty_f"])
    penalty_rows = ranked + [r for r in penalty_rows if r not in ranked]

    spread = []
    for i, cat in enumerate(cats):
        vals = [g["values"][i] for g in hour_groups
                if not g["baseline"] and g["values"][i] is not None]
        if len(vals) > 1:
            spread.append({"cat": cat, "range": max(vals) - min(vals)})
    widest = max(spread, key=lambda r: r["range"]) if spread else None
    narrowest = min(spread, key=lambda r: r["range"]) if spread else None

    hour_label = ("Bias at each of the four synoptic instants, by model — 90-day window, "
                  "00Z initialization, bilinear interpolation, lead day 1, all stations pooled")
    lead_label = ("Bias of the instantaneous 2 m temperature against lead day, one line per "
                  "model — 90-day window, 00Z initialization, bilinear interpolation")
    return {
        "hours": hours,
        "hour_cats": cats,
        "hour_when": [{"utc": f"{h:02d}Z", "local": _local_hour_span(stations, h),
                       "when": HOUR_WHEN.get(h, "")} for h in hours],
        "hour_rows": hour_rows,
        "hour_fig": Markup(svg.grouped_bias_bars(hour_groups, categories=cats, label=hour_label)),
        "hour_label": hour_label,
        "leads": leads,
        "lead_rows": lead_rows,
        "lead_fig": Markup(svg.multi_line(
            [f"d{d}" for d in leads], lead_series, label=lead_label, x_title="lead day")),
        "lead_label": lead_label,
        "penalty_rows": penalty_rows,
        "penalty_pairs": [{"sampled": a, "cli": b} for a, b in DIAGNOSTICS_PENALTY_PAIRS],
        "penalty_identity": payload["sampling_penalty"]["identity"],
        "candidates": DIAGNOSTICS_CANDIDATES,
        "lede": DIAGNOSTICS_LEDE,
        "spread": None if widest is None or narrowest is None else {
            "widest": widest["cat"], "widest_f": f"{widest['range']:.2f}",
            "narrowest": narrowest["cat"], "narrowest_f": f"{narrowest['range']:.2f}",
        },
        "window": DIAGNOSTICS_WINDOW,
        "init_hour": DIAGNOSTICS_INIT,
        "method": DIAGNOSTICS_METHOD,
        "lead": DIAGNOSTICS_LEAD,
        "n_stations": len(stations),
        "has_data": bool(hour_groups or lead_series),
    }


def _write_diagnostics(env, w, base_ctx, scores, models, model_idx, stations) -> None:
    payload = diagnostics_payload(scores, models)
    w.write("diagnostics/index.html", env.get_template("diagnostics.html").render(
        **base_ctx, **_diagnostics_view(payload, models, model_idx, stations)))


# ------------------------------------------------------------------------------------------
# /monthly/ — one automatically generated page per completed calendar month
# ------------------------------------------------------------------------------------------

#: A month is published only once it is over.  A partial month would rank models on a handful of
#: days and then silently change its numbers for four weeks, which is exactly what a dated
#: archive page must not do.
MONTHLY_VARIABLE = HEADLINE_VARIABLE
MONTHLY_LEAD = 1
#: The four common synoptic instants, so a fully observed station-day is four observations.
INSTANTS_PER_DAY = 4


def _month_days(month: str) -> int:
    start = pd.Timestamp(f"{month}-01")
    return int((start + pd.offsets.MonthEnd(1)).day)


def _complete_months(dates: pd.Series, as_of: date) -> list[str]:
    """Every ``YYYY-MM`` in ``dates`` whose last calendar day is at or before ``as_of``."""
    if dates is None or len(dates) == 0:
        return []
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if not len(d):
        return []
    months = sorted({str(m) for m in d.dt.strftime("%Y-%m")})
    cutoff = pd.Timestamp(as_of)
    return [m for m in months
            if pd.Timestamp(f"{m}-01") + pd.offsets.MonthEnd(1) <= cutoff]


def _monthly_units(errors: pd.DataFrame) -> pd.DataFrame:
    """The cross-station daily unit ``verify.score`` averages, restricted to the monthly slice.

    One row per ``(model_id, climo_date)`` with the day's mean absolute and signed error over the
    four instants and then over the stations present — the same two collapses, in the same order,
    that produce the published ``station_id="ALL"`` point estimate (METHODOLOGY §4).  It is
    recomputed here rather than read from ``scores`` because no published window is a calendar
    month; the formula is deliberately identical, and no interval is derived from it.
    """
    cols = {"station_id", "model_id", "init_hour", "lead_day", "variable", "method",
            "climo_date", "err"}
    if errors is None or len(errors) == 0 or not cols <= set(errors.columns):
        return pd.DataFrame(columns=["model_id", "climo_date", "a", "s"])
    e = errors[
        (errors["variable"] == MONTHLY_VARIABLE)
        & (errors["init_hour"].astype(int) == DEFAULT_INIT)
        & (errors["method"] == DEFAULT_METHOD)
        & (errors["lead_day"].astype(int) == MONTHLY_LEAD)
    ][["station_id", "model_id", "climo_date", "err"]].copy()
    if e.empty:
        return pd.DataFrame(columns=["model_id", "climo_date", "a", "s"])
    e["climo_date"] = pd.to_datetime(e["climo_date"]).dt.normalize()
    e["err"] = e["err"].astype(float)
    e["a"] = e["err"].abs()
    per_station = (e.groupby(["station_id", "model_id", "climo_date"], observed=True,
                             as_index=False)
                   .agg(a=("a", "mean"), s=("err", "mean")))
    per_station["month"] = per_station["climo_date"].dt.strftime("%Y-%m")
    return per_station


def _monthly_scores(units: pd.DataFrame, month: str) -> tuple[list[dict], dict | None]:
    """``(ranking rows, worst station-day)`` for one month, from the daily units."""
    if units is None or len(units) == 0:
        return [], None
    part = units[units["month"] == month]
    if part.empty:
        return [], None
    allrows = (part.groupby(["model_id", "climo_date"], observed=True, as_index=False)
               .agg(a=("a", "mean"), s=("s", "mean"), ns=("station_id", "nunique")))
    agg = (allrows.groupby("model_id", observed=True, as_index=False)
           .agg(mae=("a", "mean"), bias=("s", "mean"), n=("climo_date", "nunique"),
                n_stations=("ns", "mean")))
    rows = agg.sort_values(["mae", "model_id"]).to_dict("records")
    worst_i = part["a"].astype(float).idxmax()
    worst = part.loc[worst_i]
    return rows, {
        "station_id": str(worst["station_id"]),
        "model_id": str(worst["model_id"]),
        "date": worst["climo_date"].date().isoformat(),
        "abs_err": float(worst["a"]),
        "bias": float(worst["s"]),
    }


def _monthly_completeness(month: str, stations: list[Station], truth: pd.DataFrame | None,
                          truth_instant: pd.DataFrame | None) -> list[dict]:
    """Observed coverage of the month, as counts and a rate — never as a bare percentage."""
    days = _month_days(month)
    n_st = max(len(stations), 1)
    out = []

    def rate(name: str, got: int, expected: int, what: str) -> dict:
        return {"name": name, "got": got, "expected": expected, "what": what,
                "pct": None if expected <= 0 else 100.0 * got / expected,
                "pct_s": "—" if expected <= 0 else f"{100.0 * got / expected:.1f}%"}

    got = 0
    if truth_instant is not None and len(truth_instant) and "valid_time" in truth_instant:
        vt = pd.to_datetime(truth_instant["valid_time"], utc=True, errors="coerce")
        sel = truth_instant[vt.dt.strftime("%Y-%m") == month]
        got = int(sel[["station_id", "valid_time"]].drop_duplicates().shape[0]) if len(sel) else 0
    out.append(rate("Instantaneous observations", got, days * n_st * INSTANTS_PER_DAY,
                    "one routine METAR per station at each of 00/06/12/18 UTC"))

    got = 0
    if truth is not None and len(truth) and "climo_date" in truth:
        cd = pd.to_datetime(truth["climo_date"], errors="coerce")
        sel = truth[cd.dt.strftime("%Y-%m") == month]
        got = int(sel[["station_id", "climo_date"]].drop_duplicates().shape[0]) if len(sel) else 0
    out.append(rate("Daily-extreme reports", got, days * n_st,
                    "one NWS Daily Climate Report (or CF6 fallback) per station-day"))
    return out


def _monthly_qc(month: str, truth: pd.DataFrame | None,
                truth_instant: pd.DataFrame | None) -> tuple[list[dict], int]:
    """Every QC flag raised on an observation in the month, counted by flag."""
    counts: dict[str, int] = {}
    for frame, col, kind in ((truth, "climo_date", "daily extreme"),
                            (truth_instant, "valid_time", "instant")):
        if frame is None or len(frame) == 0 or col not in frame or "qc_flag" not in frame:
            continue
        when = pd.to_datetime(frame[col], utc=(col == "valid_time"), errors="coerce")
        sel = frame[when.dt.strftime("%Y-%m") == month]
        if not len(sel):
            continue
        flags = sel["qc_flag"].fillna("").astype(str)
        for value in flags[flags != ""]:
            for one in str(value).split(";"):
                one = one.strip()
                if one:
                    counts[f"{one} ({kind})"] = counts.get(f"{one} ({kind})", 0) + 1
    rows = [{"flag": k, "n": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return rows, sum(counts.values())


def _version_events(daily: pd.DataFrame | None) -> dict[str, list[dict]]:
    """``YYYY-MM -> [{model_id, model_version, first_day}]``: where a version segment begins.

    A model version that first appears mid-month is an upstream change, and scores either side of
    it are not pooled (METHODOLOGY §9).  The first appearance of every model is itself an event —
    the month a system entered the record.
    """
    out: dict[str, list[dict]] = {}
    if daily is None or len(daily) == 0 or "model_version" not in daily:
        return out
    d = daily[["model_id", "model_version", "climo_date"]].copy()
    d["model_version"] = d["model_version"].fillna("unknown").astype(str)
    d["climo_date"] = pd.to_datetime(d["climo_date"], errors="coerce")
    d = d[d["climo_date"].notna()]
    if d.empty:
        return out
    first = (d.groupby(["model_id", "model_version"], observed=True, as_index=False)
             .agg(first_day=("climo_date", "min")))
    for r in first.to_dict("records"):
        month = r["first_day"].strftime("%Y-%m")
        out.setdefault(month, []).append({
            "model_id": r["model_id"], "model_version": r["model_version"],
            "first_day": r["first_day"].date().isoformat(),
        })
    for rows in out.values():
        rows.sort(key=lambda r: (r["first_day"], r["model_id"]))
    return out


def _monthly_pages(errors, truth, truth_instant, daily, stations, model_idx,
                   as_of: date) -> list[dict]:
    """One rendering context per completed month, newest first."""
    units = _monthly_units(errors)
    months = _complete_months(units["climo_date"] if len(units) else pd.Series(dtype="datetime64[ns]"),
                              as_of)
    if not months:
        return []
    names = _display_map(model_idx)
    versions = _version_events(daily)
    pages = []
    for month in sorted(months, reverse=True):
        rows, worst = _monthly_scores(units, month)
        if not rows:
            continue
        days = _month_days(month)
        ranked = [r for r in rows if r["model_id"] != PERSISTENCE_ID]
        table = []
        for r in rows:
            baseline = r["model_id"] == PERSISTENCE_ID
            table.append({
                "rank": "—" if baseline else str(
                    [x["model_id"] for x in ranked].index(r["model_id"]) + 1),
                "id": r["model_id"], "name": names.get(r["model_id"], r["model_id"]),
                "baseline": baseline,
                "mae": f_delta(r["mae"]), "mae_f": float(r["mae"]) * C_TO_F_DELTA,
                "bias": f_signed(r["bias"]), "bias_short": _BIAS_SHORT.get(
                    svg.bias_class(float(r["bias"]) * C_TO_F_DELTA), "fl"),
                "n": int(r["n"]), "days": days,
                # A month is at most 31 days, so no row here clears the site's MIN_N bar by much;
                # a row scored on fewer than all of the month's days is greyed the same way a
                # short window is greyed elsewhere, and its coverage is printed beside it.
                "low_n": int(r["n"]) < days,
                "coverage": f"{100.0 * int(r['n']) / days:.0f}%",
                "n_stations": f"{float(r['n_stations']):.1f}",
                "permalink": permalink_url(ALL_STATIONS, r["model_id"], MONTHLY_LEAD),
            })
        qc_rows, qc_total = _monthly_qc(month, truth, truth_instant)
        if worst is not None:
            worst = {**worst,
                     "name": names.get(worst["model_id"], worst["model_id"]),
                     "abs_err_f": _minus(f"{worst['abs_err'] * C_TO_F_DELTA:.2f}"),
                     "bias_f": _minus(f"{worst['bias'] * C_TO_F_DELTA:+.2f}"),
                     "permalink": permalink_url(worst["station_id"], worst["model_id"],
                                                MONTHLY_LEAD)}
        pages.append({
            "month": month,
            "month_label": _MONTH_LABEL(f"{month}-01"),
            "days": days,
            "rows": table,
            "n_models": len(ranked),
            "leader": table[0] if table and not table[0]["baseline"] else None,
            "completeness": _monthly_completeness(month, stations, truth, truth_instant),
            "qc_rows": qc_rows,
            "qc_total": qc_total,
            "worst": worst,
            "versions": versions.get(month, []),
            "variable": MONTHLY_VARIABLE,
            "variable_label": VAR_LABEL.get(MONTHLY_VARIABLE, MONTHLY_VARIABLE),
            "lead": MONTHLY_LEAD,
            "n_stations": len(stations),
        })
    return pages


def _write_monthly(env, w, base_ctx, pages: list[dict]) -> int:
    """The index and one page per completed month.  Returns the number of month pages."""
    index = [{"month": p["month"], "month_label": p["month_label"], "days": p["days"],
              "n_models": p["n_models"], "leader": p["leader"], "qc_total": p["qc_total"],
              "n_versions": len(p["versions"]),
              "completeness": p["completeness"][0]["pct_s"] if p["completeness"] else "—"}
             for p in pages]
    w.write("monthly/index.html", env.get_template("monthly_index.html").render(
        **base_ctx, months=index))
    for i, page in enumerate(pages):
        w.write(f"monthly/{page['month']}/index.html",
                env.get_template("monthly.html").render(
                    **base_ctx, **page,
                    newer=index[i - 1] if i > 0 else None,
                    older=index[i + 1] if i + 1 < len(index) else None))
    return len(pages)


def _monthly_feed_entries(pages: list[dict]) -> list[dict]:
    """One Atom entry per month page, published at the first publish slot after the month ended."""
    out = []
    for p in pages:
        leader = p["leader"]
        head = (f"{leader['name']} had the lowest MAE among the models scored that month at lead "
                f"day {p['lead']} ({leader['mae']} °F over {leader['n']} of {p['days']} days)."
                if leader else "No model reached a full month of scored days.")
        stamp = (pd.Timestamp(f"{p['month']}-01") + pd.offsets.MonthEnd(1)
                 + pd.Timedelta(days=1)).strftime(f"%Y-%m-%dT{PUBLISH_HOUR_UTC:02d}:00:00+00:00")
        out.append({
            "date": p["month"],
            "title": f"CastCheck monthly review {p['month_label']}",
            "href": f"{SITE_URL}/monthly/{p['month']}/",
            "id": f"tag:castcheck,{p['month']}:monthly",
            "updated": stamp,
            "summary": (f"{p['month_label']}: {head} {p['n_models']} systems scored across "
                        f"{p['n_stations']} stations; {p['qc_total']} QC flag"
                        f"{'' if p['qc_total'] == 1 else 's'} on the month's observations; "
                        f"{len(p['versions'])} upstream version change"
                        f"{'' if len(p['versions']) == 1 else 's'}."),
        })
    return out


def _write_indexes(env, w, base_ctx, scores, stations, model_idx) -> None:
    """``/stations/`` and ``/models/``: the two directories the top navigation points at.

    Both are built from the registry, so they exist and are complete even before a single day has
    been scored; the sample sizes come from the default view and are simply blank until then.
    """
    names = _display_map(model_idx)
    head = pd.DataFrame()
    variable = _pick_series_variable(scores)
    if scores is not None and len(scores):
        sc = scores
        head = sc[(sc["window"] == DEFAULT_WINDOW) & (sc["init_hour"].astype(int) == DEFAULT_INIT)
                  & (sc["method"] == DEFAULT_METHOD) & (sc["variable"] == variable)
                  & (sc["lead_day"].astype(int) == 1)]

    # A4: the persistence baseline is not a forecast system, and counting its days here made every
    # station read 90 scored days while the model pages said 16–28. Only real models count.
    forecast_ids = {mid for mid, info in model_idx.items() if not info.get("baseline")}
    head_models = head[head["model_id"].isin(forecast_ids)] if len(head) else head

    st_rows = []
    for st in stations:
        part = head_models[head_models["station_id"] == st.id] if len(head_models) else head_models
        n = int(part["n"].max()) if len(part) else 0
        dz = _station_dz(st)
        grid = _station_field(st, "grid_elev_m")
        lapse = _station_lapse(st)
        st_rows.append({
            "id": st.id, "name": st.name, "cli_pil": st.cli_pil, "tz": st.tz,
            "offset": f"{st.std_offset_h:+d}",
            "elev": "—" if st.elev_m is None else f"{st.elev_m:.0f} m",
            "grid_elev": "—" if grid is None else f"{float(grid):.0f} m",
            "dz": "—" if dz is None else f"{dz:+.0f} m",
            "lapse": "—" if lapse is None else f"{lapse:.1f} K",
            "market_city": _station_field(st, "market_city") or "—",
            "n": n or "—", "n_models": int(part["model_id"].nunique()) if len(part) else "—",
            "low_n": n < MIN_N,
        })

    # The shared availability scale the per-card progress bars are drawn against.
    span_lo = span_hi = None
    if scores is not None and len(scores):
        allrows = scores[scores["window"] == "all"]
        if len(allrows):
            span_lo = pd.to_datetime(allrows["period_start"], errors="coerce").min()
            span_hi = pd.to_datetime(allrows["period_end"], errors="coerce").max()
    span_days = max(((span_hi - span_lo).days if span_lo is not None
                     and span_hi is not None else 0), 1)

    md_rows = []
    for mid in sorted(model_idx, key=lambda k: (model_idx[k].get("baseline", False), names[k])):
        info = model_idx[mid]
        part = head[(head["model_id"] == mid) & (head["station_id"] == ALL_STATIONS)] \
            if len(head) else head
        allw = pd.DataFrame()
        if scores is not None and len(scores):
            allw = scores[(scores["model_id"] == mid) & (scores["window"] == "all")]
        n = int(part["n"].max()) if len(part) else 0
        # MAE across lead days, on the default view: the shape of the card's sparkline.
        spark = []
        sp_src = pd.DataFrame()
        if scores is not None and len(scores):
            sp_src = scores[(scores["model_id"] == mid) & (scores["station_id"] == ALL_STATIONS)
                            & (scores["window"] == DEFAULT_WINDOW)
                            & (scores["init_hour"].astype(int) == DEFAULT_INIT)
                            & (scores["method"] == DEFAULT_METHOD)
                            & (scores["variable"] == variable)]
        for ld in SPARK_LEADS:
            hit = sp_src[sp_src["lead_day"].astype(int) == ld] if len(sp_src) else sp_src
            spark.append(_f(hit["mae"].iloc[0]) if len(hit) else None)
        left = width = 0.0
        p_start = p_end = ""
        if len(allw):
            s0 = pd.to_datetime(allw["period_start"], errors="coerce").min()
            e0 = pd.to_datetime(allw["period_end"], errors="coerce").max()
            if pd.notna(s0) and pd.notna(e0) and span_lo is not None:
                p_start, p_end = s0.date().isoformat(), e0.date().isoformat()
                left = round(100.0 * (s0 - span_lo).days / span_days, 2)
                width = round(min(max(100.0 * ((e0 - s0).days + 1) / span_days, 1.0),
                                  100.0 - left), 2)
        md_rows.append({
            "model_id": mid, "name": names[mid], "source": info.get("source", ""),
            "product": info.get("product", ""), "init_field": info.get("init_field"),
            "baseline": info.get("baseline", False),
            "n": n or "—", "n_raw": n, "low_n": n < MIN_N,
            "period": f_period(allw["period_start"].min(), allw["period_end"].max())
            if len(allw) else "—",
            "period_start": p_start or "—", "period_end": p_end or "—",
            "avail_left": left, "avail_width": width,
            "spark_svg": Markup(svg.sparkline(
                spark, label=f"{names[mid]} MAE by lead day "
                             f"{SPARK_LEADS[0]} to {SPARK_LEADS[-1]}", width=112.0,
                muted=n < MIN_N)),
            "has_spark": any(v is not None for v in spark),
            **_segment_note(allw),
        })

    shared = {"window_label": f"the last {DEFAULT_WINDOW[:-1]} days",
              "init_hour": f"{DEFAULT_INIT:02d}", "method": DEFAULT_METHOD,
              "variable": variable, "variable_label": VAR_LABEL.get(variable, variable),
              "spark_leads": list(SPARK_LEADS),
              "avail_start": span_lo.date().isoformat() if span_lo is not None else "—",
              "avail_end": span_hi.date().isoformat() if span_hi is not None else "—"}
    w.write("stations/index.html",
            env.get_template("stations.html").render(**base_ctx, **shared, rows=st_rows))
    w.write("models/index.html",
            env.get_template("models.html").render(**base_ctx, **shared, rows=md_rows))


def _write_api_index(env, w, base_ctx, api_written) -> None:
    w.write("api/v1/index.html", env.get_template("api.html").render(
        **base_ctx, api_written=api_written or {}))


def _feed_entries(sc: pd.DataFrame, model_idx: dict, as_of_s: str, built_at: str) -> list[dict]:
    """One entry per published day found in ``data/scores/history`` (newest first)."""
    dates = []
    hist = REPO_ROOT / "data" / "scores" / "history"
    if hist.exists():
        # A conflict copy ("2026-08-30 2.parquet") once put an unparseable <updated> in the feed
        # and broke every reader.  Only a bare ISO date is a publication date.
        dates = sorted((p.stem for p in hist.glob("*.parquet") if _HISTORY_DATE.fullmatch(p.stem)),
                       reverse=True)[:30]
    if as_of_s not in dates:
        dates = [as_of_s, *dates]
    variable = _pick_series_variable(sc)
    board = sc[(sc["station_id"] == ALL_STATIONS) & (sc["window"] == DEFAULT_WINDOW)
               & (sc["init_hour"] == DEFAULT_INIT) & (sc["method"] == DEFAULT_METHOD)
               & (sc["variable"] == variable) & (sc["lead_day"] == 1)
               & (sc["model_id"] != PERSISTENCE_ID)]
    board = board[board["n"] >= MIN_N] if len(board) else board
    if len(board):
        top = board.sort_values("mae").iloc[0]
        summary = (f"Lead day 1, {VAR_LABEL.get(variable, variable)}, {DEFAULT_WINDOW} window, "
                   f"{DEFAULT_INIT:02d}Z, {DEFAULT_METHOD}: lowest MAE among ranked models "
                   f"(n >= {MIN_N}) is "
                   f"{f_delta(top['mae'])} °F ({_mname(model_idx, top['model_id'])}, "
                   f"n = {int(top['n'])}).")
    else:
        summary = (f"No group in the default view has reached {MIN_N} scored days yet, so nothing "
                   f"is ranked; every number is published with its sample size.")
    return [{"date": d,
             "title": f"CastCheck update {d}",
             "href": f"{SITE_URL}/",
             "id": f"tag:castcheck,{d}:update",
             "summary": summary if i == 0 else
             "Scores recomputed from scratch for this publication date.",
             # the newest entry is stamped with the actual build, older ones with their slot
             "updated": built_at if i == 0 else f"{d}T{PUBLISH_HOUR_UTC:02d}:00:00+00:00"}
            for i, d in enumerate(dates[:30])]


def _write_feed(env, w, base_ctx, entries) -> None:
    w.write("feed.xml", env.get_template("feed.xml").render(
        **base_ctx, entries=entries or []), count=False)
