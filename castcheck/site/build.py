"""Static site generator (DESIGN §6).

Jinja2 → ``public/``.  Every page is readable with JavaScript disabled: the tables *are* the
content, every figure is server-rendered inline SVG (``site/svg.py``) with an equivalent table
beside it, and ``assets/chart.js`` only adds the theme toggle and a hover read-out.

Every analysis choice is a URL, not a widget.  The four dimensions — window (30d/90d/365d/all),
initialization (00Z/12Z), interpolation (bilinear/nearest) and variable (tmax/tmin) — are baked
into static paths so that any view can be linked, cited and diffed:

``/``                                           the default view: 90d · 00Z · bilinear · tmax
``/v/{window}-{init}z-{method}-{variable}/``     each of the 32 leaderboard views (``/`` is a copy)
``/station/{ICAO}/``  ``/station/{ICAO}/v/{window}-{init}z-{method}/``
``/model/{model_id}/``  ``/model/{model_id}/v/{window}-{init}z-{method}/``
``/station/{ICAO}/model/{model_id}/lead/{d}/``  the permanent link (DESIGN §6, fixed for good)
``/methodology/``  ``/status/``  ``/data/``  ``/api/v1/``  ``/feed.xml``  ``/_headers``

``station_id="ALL"`` is published as a pseudo-station so the cross-station aggregate has permanent
links too.  Errors are stored in °C (METHODOLOGY §3) and displayed in °F throughout.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .. import METHODOLOGY_VERSION, SCHEMA_VERSION, __version__
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

__all__ = [
    "COLUMN_DOCS",
    "FAIRNESS",
    "FAIRNESS_BANNER",
    "SITE_URL",
    "VIEWS",
    "build_site",
    "citation",
    "citation_long",
    "next_update",
    "view_slug",
]

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
ASSETS = HERE / "assets"

SITE_URL = "https://castcheck.zifanzhang.com"
REPO_URL = "https://github.com/zifanzhang/castcheck"
HF_URL = "https://huggingface.co/datasets/castcheck/temperature-verification"

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
VARIABLES = ("tmax", "tmin")
DEFAULT_WINDOW = "90d"
DEFAULT_INIT = 0
DEFAULT_METHOD = "bilinear"
DEFAULT_VARIABLE = "tmax"
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
VAR_LABEL = {"tmax": "daily maximum", "tmin": "daily minimum"}

#: /data/ schema table.  column → (type, unit, meaning)
COLUMN_DOCS: dict[str, tuple[str, str, str]] = {
    "station_id": ("string", "—", "ICAO identifier, or ALL for the cross-station aggregate"),
    "model_id": ("string", "—", "stable model identifier from config/models.yaml"),
    "model_version": ("string", "—", "upstream cycle/version string as advertised by the producer"),
    "init_hour": ("int8", "UTC hour", "model initialization hour, 0 or 12"),
    "init_time": ("timestamp", "UTC", "model initialization time"),
    "valid_time": ("timestamp", "UTC", "forecast valid time"),
    "lead_h": ("int16", "hours", "valid_time − init_time"),
    "lead_day": ("int8", "days", "target climatological date − UTC date of the initialization"),
    "variable": ("string", "—", "tmax or tmin (daily extreme of the four common samples)"),
    "bucket_h": ("int8", "hours", "accumulation window of a native extreme field, 0 if instantaneous"),
    "method": ("string", "—", "grid-to-station interpolation: bilinear (headline) or nearest"),
    "window": ("string", "—", "scoring window: 30d, 90d, 365d or all"),
    "n": ("int32", "days", "number of scored climatological days in the window"),
    "n_stations": ("float32", "stations", "mean number of stations behind each day of an ALL row "
                   "(1 for a single-station row)"),
    "n_flagged": ("int32", "days", "how many of those days carry a QC flag on the observation"),
    "n_common": ("int32", "days", "days on which both models of the pair have a forecast"),
    "n_samples": ("int8", "count", "how many of the four common samples were present (0–4)"),
    "mae": ("float32", "°C", "mean absolute error, forecast − observed"),
    "bias": ("float32", "°C", "mean signed error; positive = model too warm"),
    "rmse": ("float32", "°C", "root mean squared error"),
    "hit1f": ("float32", "fraction", "share of days with |error| ≤ 1 °F"),
    "hit2f": ("float32", "fraction", "share of days with |error| ≤ 2 °F"),
    "hit3f": ("float32", "fraction", "share of days with |error| ≤ 3 °F"),
    "mae_debiased": ("float32", "°C", "MAE after removing the per-station constant bias of the "
                     "window; the part of the error that is not a fixed offset"),
    "skill_persistence": ("float32", "fraction", "1 − MAE/MAE(persistence); positive is better"),
    "skill_persistence_debiased": ("float32", "fraction",
                                   "the same skill score computed on the debiased errors, so a "
                                   "station with a large constant offset is not scored as skill-less"),
    "mae_ci_low": ("float32", "°C", "2.5th percentile, moving-block bootstrap"),
    "mae_ci_high": ("float32", "°C", "97.5th percentile, moving-block bootstrap"),
    "bias_ci_low": ("float32", "°C", "2.5th percentile of the bias bootstrap"),
    "bias_ci_high": ("float32", "°C", "97.5th percentile of the bias bootstrap"),
    "rmse_ci_low": ("float32", "°C", "2.5th percentile of the RMSE bootstrap"),
    "rmse_ci_high": ("float32", "°C", "97.5th percentile of the RMSE bootstrap"),
    "hit1f_ci_low": ("float32", "fraction", "2.5th percentile of the ±1 °F hit-rate bootstrap"),
    "hit1f_ci_high": ("float32", "fraction", "97.5th percentile of the ±1 °F hit-rate bootstrap"),
    "segment_start": ("date", "LST day", "first day of the current model-version segment; scores "
                      "cover only this segment"),
    "mae_diff": ("float32", "°C", "MAE(model_a) − MAE(model_b) on their common days"),
    "ci_low": ("float32", "°C", "2.5th percentile of the paired bootstrap difference"),
    "ci_high": ("float32", "°C", "97.5th percentile of the paired bootstrap difference"),
    "significant": ("bool", "—", "true when the 95 % interval of the difference excludes zero"),
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
    "schema_version": ("string", "—", "data-model version (DESIGN §3)"),
    "methodology_version": ("string", "—", "METHODOLOGY version that produced the numbers"),
}

CHANGELOG = [
    ("0.1", "2026-08-30", "First public build: 2 m temperature, 23 stations, ECMWF IFS HRES, "
                          "NCEP GFS and the NOAA/CIRA AIWP models, persistence baseline, "
                          "moving-block bootstrap intervals."),
]

LIMITATIONS = [
    "Daily extremes are the max/min of four 6-hourly samples, so they under-state the true "
    "afternoon peak and over-state the pre-dawn trough. The bias is identical for every model and "
    "is therefore fair for comparison, but the absolute errors are not what a user of a "
    "post-processed forecast experiences (METHODOLOGY §2.3).",
    "No elevation or lapse-rate correction is applied; stations whose elevation differs sharply "
    "from the 0.25° grid cell carry a representativeness error that is charged to the model.",
    "Truth is the first final NWS CLI report. Later corrections are stored but never change a "
    "published score, so a corrected observation leaves a permanent, documented discrepancy.",
    "Models enter the record on different dates, so their windows are not identical. Pairwise "
    "comparisons are computed on common days only; the leaderboard columns are not.",
    "Groups with fewer than 30 scored days are published but greyed out and excluded from every "
    "ranking; early in a model's record most windows are in that state.",
    "The 0.25° AIWP archive is a research product; outages there appear here as gaps, not as bad "
    "forecasts (see /status/).",
]


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


def f_delta(c, digits: int = 2) -> str:
    return "—" if _isnan(c) else f"{float(c) * C_TO_F_DELTA:.{digits}f}"


def f_signed(c, digits: int = 2) -> str:
    return "—" if _isnan(c) else f"{float(c) * C_TO_F_DELTA:+.{digits}f}"


def f_ci(lo, hi, digits: int = 2) -> str:
    """The compact interval notation of docs/05 §A: ``[1.9, 2.3]``."""
    if _isnan(lo) or _isnan(hi):
        return "—"
    return f"[{float(lo) * C_TO_F_DELTA:.{digits}f}, {float(hi) * C_TO_F_DELTA:.{digits}f}]"


def f_pct(x) -> str:
    return "—" if _isnan(x) else f"{float(x) * 100:.0f}%"


def f_skill(x) -> str:
    return "—" if _isnan(x) else f"{float(x):+.2f}"


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
        f"{SITE_URL}{permalink_url(station_id, model_id, lead)}"
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


def _row_view(r, station_id: str, model_id: str, lead: int, model_idx: dict | None = None) -> dict:
    bias_f = _f(r["bias"])
    bias_sig = not (_isnan(r["bias_ci_low"]) or _isnan(r["bias_ci_high"])) and not (
        float(r["bias_ci_low"]) <= 0.0 <= float(r["bias_ci_high"])
    )
    n_stations = r.get("n_stations")
    n_flagged = r.get("n_flagged")
    seg = r.get("segment_start")
    return {
        "model_name": _mname(model_idx, model_id),
        "mae_debiased": f_delta(r.get("mae_debiased")),
        "skill_debiased": f_skill(r.get("skill_persistence_debiased")),
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
        "variable": r["variable"],
        "init_hour": f"{int(r['init_hour']):02d}",
        "method": r["method"],
        "window": r["window"],
        "n": int(r["n"]),
        "low_n": int(r["n"]) < MIN_N,
        "mae": f_delta(r["mae"]),
        "mae_f": _f(r["mae"]),
        "mae_ci": f_ci(r["mae_ci_low"], r["mae_ci_high"]),
        "bias": f_signed(r["bias"]),
        "bias_f": bias_f,
        "bias_ci": f_ci(r["bias_ci_low"], r["bias_ci_high"]),
        "bias_class": svg.bias_class(bias_f, bias_sig),
        "bias_significant": bias_sig,
        "rmse": f_delta(r["rmse"]),
        "hit1f": f_pct(r["hit1f"]),
        "hit2f": f_pct(r["hit2f"]),
        "hit3f": f_pct(r["hit3f"]),
        "skill": f_skill(r["skill_persistence"]),
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


_WINDOW_OPTS = [(w, "all history" if w == "all" else f"last {w[:-1]} days") for w in WINDOWS]
_INIT_OPTS = [(i, f"{i:02d}Z") for i in INITS]
_METHOD_OPTS = [(m, m) for m in METHODS]
_VAR_OPTS = [(v, "daily max" if v == "tmax" else "daily min") for v in VARIABLES]

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


class _Writer:
    def __init__(self, out: Path):
        self.out = out
        self.n = 0

    def write(self, relpath: str, html: str, count: bool = True) -> Path:
        path = self.out / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
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
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
    status_report: dict | None = None,
    api_written: dict[str, int] | None = None,
    permalinks: bool = True,
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

    scores = scores if scores is not None else pd.DataFrame(columns=SCORE_COLUMNS)
    pairwise = pairwise if pairwise is not None else pd.DataFrame(columns=PAIRWISE_COLUMNS)

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
        "next_update": next_update(built_at),
        "min_n": MIN_N,
        "site_url": SITE_URL,
        "repo_url": REPO_URL,
        "hf_url": HF_URL,
    }

    # Variant directories are named after the view slugs; if the set of views ever changes, the
    # old pages would linger in public/ and be deployed as stale duplicates. They are cheap to
    # regenerate, so drop them first.
    _prune_variants(out)

    # assets and the Cloudflare Pages header rules
    asset_urls = _write_assets(out)
    base_ctx.update(asset_urls)
    _write_headers(out)

    counts = {"pages": 0, "stations": 0, "models": 0, "permalinks": 0, "leaderboards": 0}

    # ---- always-present pages -------------------------------------------------------------
    _write_methodology(env, w, base_ctx)
    _write_status(env, w, base_ctx, status_report, _display_map(_model_index(models)))
    _write_indexes(env, w, base_ctx, scores, stations, _model_index(models))
    _write_api_index(env, w, base_ctx, api_written)

    empty = scores is None or len(scores) == 0
    errors = pd.DataFrame()
    if not empty and daily is not None and truth is not None and len(daily) and len(truth):
        try:
            errors = error_table(daily, truth)
        except Exception:  # noqa: BLE001 - a malformed shard must not break the build
            errors = pd.DataFrame()
    downloads = _write_downloads(out, scores, pairwise, errors, stations, models)
    _write_data(env, w, base_ctx, api_written, scores, pairwise, downloads)

    if empty:
        w.write("index.html", env.get_template("empty.html").render(
            **base_ctx,
            heading="CastCheck — no scores yet",
            message="No forecast has been matched to an observation yet, so there is nothing to "
                    "rank. This page will fill in automatically once the first climatological day "
                    "has both a model forecast and its NWS Daily Climate Report.",
        ))
        _write_feed(env, w, base_ctx, None)
        counts["pages"] = w.n
        return counts

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

    series_idx, month_idx, allday_idx = _error_indices(errors)

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
        avail_all = sub[(sub["window"] == "all") & (sub["variable"] == DEFAULT_VARIABLE)
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

    _write_feed(env, w, base_ctx, _feed_entries(sc, model_idx, as_of_s, built_at))
    counts["pages"] = w.n
    return counts


# ------------------------------------------------------------------------------------------
# error-series indices (built once, read by every permanent link)
# ------------------------------------------------------------------------------------------

def _error_indices(errors: pd.DataFrame):
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
        & (errors["variable"] == DEFAULT_VARIABLE)
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

    series: dict = {}
    allday: dict = {}
    cutoff = e["climo_date"].max() - pd.Timedelta(days=SERIES_DAYS - 1)
    for key, grp in e.groupby(["station_id", "model_id", "lead_day"], observed=True):
        k = (key[0], key[1], int(key[2]))
        allday[k] = grp["err_f"].tolist()
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

    matrix = _lead_matrix(sub, model_idx, leads)
    avail = sc[
        (sc["station_id"] == ALL_STATIONS) & (sc["window"] == "all")
        & (sc["variable"] == variable) & (sc["init_hour"] == int(init_hour))
        & (sc["method"] == method) & (sc["lead_day"] == 1)
    ]
    availability, avail_start, avail_end = _availability(avail, model_idx)
    map_points, map_rows = _map_points(sc, station_idx, view, model_idx)

    return env.get_template("index.html").render(
        **base_ctx,
        boards=boards,
        spark_leads=list(SPARK_LEADS),
        headline=headline,
        matrix=matrix,
        window=window,
        window_label="all available history" if window == "all" else f"the last {window[:-1]} days",
        init_hour=f"{int(init_hour):02d}",
        method=method,
        variable=variable,
        variable_label=VAR_LABEL[variable],
        n_stations=n_stations,
        variants=_variant_links("/", view, _DIMS4),
        availability=availability,
        avail_start=avail_start,
        avail_end=avail_end,
        station_links=station_links,
        map_svg=Markup(svg.us_map(
            map_points,
            label=(f"Mean bias in °F of each station's best model, lead day 1, "
                   f"{VAR_LABEL[variable]}, {window} window, {int(init_hour):02d}Z, {method}"))),
        map_rows=map_rows,
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
        try:
            s = float(r["skill"])
        except (TypeError, ValueError):
            continue
        best_skill = s if best_skill is None else max(best_skill, s)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["mark"] = "★" if r["model_id"] == leader else marks.get(r["model_id"], "·")
        r["mark_title"] = _MARK_TITLE.get(r["mark"], "")
        r["best_mae"] = best_mae is not None and r["mae_f"] is not None and abs(r["mae_f"] - best_mae) < 1e-9
        r["best_skill"] = (best_skill is not None and r["skill"] not in ("—",)
                           and abs(float(r["skill"]) - best_skill) < 1e-9)
    for r in others:
        r["rank"] = None
        r["mark"] = ""
        r["mark_title"] = ""
        r["best_mae"] = r["best_skill"] = False
    return {"lead": lead, "rows": ranked + others, "leader": leader,
            "leader_name": _mname(model_idx, leader) if leader else "",
            "n_ranked": len(ranked), "any_ranked": bool(ranked)}


_MARK_TITLE = {
    "★": "lowest MAE in this view",
    "=": "not distinguishable from the leader (95 % interval of the paired difference includes 0)",
    "▼": "significantly worse than the leader (95 % interval excludes 0)",
    "▲": "significantly better than the leader on their common days",
    "·": "no paired comparison available",
}


def _significance(pw_sub: pd.DataFrame, lead: int, leader: str | None) -> dict[str, str]:
    """Symbol per model: is its MAE distinguishable from the leader's on their common days?"""
    if leader is None or pw_sub is None or len(pw_sub) == 0:
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
        if not bool(r["significant"]):
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


def _map_points(sc: pd.DataFrame, station_idx: dict, view,
                model_idx: dict | None = None) -> tuple[list[dict], list[dict]]:
    """One dot per station: bias of the leading model at lead day 1 in this view."""
    window, init_hour, method, variable = view
    sub = sc[(sc["station_id"] != ALL_STATIONS) & (sc["window"] == window)
             & (sc["init_hour"] == int(init_hour)) & (sc["method"] == method)
             & (sc["variable"] == variable) & (sc["lead_day"] == 1)
             & (sc["model_id"] != PERSISTENCE_ID)]
    if sub.empty:
        return [], []
    points, rows = [], []
    for sid, grp in sub.groupby("station_id", observed=True):
        best = grp.sort_values("mae").iloc[0]
        st = station_idx.get(sid)
        bias_f = _f(best["bias"])
        sig = not (_isnan(best["bias_ci_low"]) or _isnan(best["bias_ci_high"])) and not (
            float(best["bias_ci_low"]) <= 0.0 <= float(best["bias_ci_high"]))
        row = {
            "id": sid, "name": st.name if st else sid,
            "lat": st.lat if st else None, "lon": st.lon if st else None,
            "n": int(best["n"]), "model_id": best["model_id"],
            "model_name": _mname(model_idx, best["model_id"]),
            "mae": f_delta(best["mae"]), "bias": f_signed(best["bias"]),
            "bias_class": svg.bias_class(bias_f, sig),
            "sign": "" if bias_f is None else ("+" if bias_f > 0 else "−"),
            "href": f"/station/{sid}/",
            "low_n": int(best["n"]) < MIN_N,
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
    blocks = []
    for var in VARIABLES:
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
                        "bias_class": svg.bias_class(bias_f), "n": int(r["n"]),
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
        blocks.append({"variable": var, "rows": rows, "flagged": flagged})
    ctx = dict(base_ctx)
    ctx.update(extra or {})
    return env.get_template(template).render(
        **ctx, blocks=blocks, leads=leads, spark_leads=list(SPARK_LEADS),
        window=window,
        window_label="all available history" if window == "all" else f"the last {window[:-1]} days",
        init_hour=f"{int(init_hour):02d}", method=method,
    )


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
        out.append({
            "other": other,
            "other_name": _mname(model_idx, other),
            "n_common": int(r["n_common"]),
            "diff": f_signed(sign * r["mae_diff"]),
            "ci": f_ci(lo, hi),
            "significant": bool(r["significant"]),
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

def _write_permalinks(env, w, out, base_ctx, sc, pw, model_idx, station_idx, series_idx,
                      month_idx, allday_idx, as_of_s, data_through) -> int:
    pw_head = pd.DataFrame()
    if len(pw):
        pw_head = pw[
            (pw["window"] == DEFAULT_WINDOW)
            & (pw["init_hour"] == DEFAULT_INIT)
            & (pw["method"] == DEFAULT_METHOD)
            & (pw["variable"] == DEFAULT_VARIABLE)
        ]
    pw_idx: dict[tuple[str, int], pd.DataFrame] = {}
    if len(pw_head):
        for key, grp in pw_head.groupby(["station_id", "lead_day"], observed=True):
            pw_idx[(key[0], int(key[1]))] = grp

    tpl = env.get_template("permalink.html")
    n = 0
    for (sid, mid, lead), grp in sc.groupby(["station_id", "model_id", "lead_day"], observed=True):
        lead = int(lead)
        grp = grp.assign(_w=grp["window"].map(WINDOW_ORDER).fillna(9))
        rows = [_row_view(r, sid, mid, lead, model_idx) for _, r in
                grp.sort_values(["variable", "init_hour", "method", "_w"]).iterrows()]
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
                   f"{DEFAULT_VARIABLE}")))
        hist_svg, hist_rows = svg.histogram(
            allday_idx.get(key, []),
            label=(f"Distribution of the signed daily error in °F for {_mname(model_idx, mid)} "
                   f"at {sid}, lead day {lead}"))
        series_rows = []
        if ser:
            series_rows = [
                {"date": d, "err": "—" if v is None else f"{v:+.2f}"}
                for d, v in zip(ser["dates"], ser["values"])
            ][::-1]
        base = f"station/{sid}/model/{mid}/lead/{lead}"
        _write_permalink_csv(out / base / "errors.csv", sid, mid, lead, ser)
        html = tpl.render(
            **base_ctx,
            station_id=sid, model_id=mid, lead=lead,
            station_name=st_name,
            station=st,
            model_name=_mname(model_idx, mid),
            model_family=model_idx.get(mid, {}).get("family", mid),
            model_info=model_idx.get(mid, {}),
            segment=_segment_note(grp),
            n_flagged=max((r["n_flagged"] for r in rows), default=0),
            rows=rows, pairwise=pairs, pairwise_window=DEFAULT_WINDOW,
            chart_init=f"{DEFAULT_INIT:02d}", chart_method=DEFAULT_METHOD,
            chart_variable=DEFAULT_VARIABLE, chart_variable_label=VAR_LABEL[DEFAULT_VARIABLE],
            chart_window=DEFAULT_WINDOW,
            series_days=SERIES_DAYS,
            series_svg=chart, series_rows=series_rows,
            hist_svg=Markup(hist_svg), hist_rows=hist_rows,
            months=month_idx.get(key, [])[::-1][:24],
            availability=_availability(grp[(grp["window"] == "all")
                                           & (grp["variable"] == DEFAULT_VARIABLE)],
                                       model_idx)[0],
            json_url=f"/api/v1/scores/{sid}/{mid}/{lead}.json",
            csv_url=f"/{base}/errors.csv",
            citation=citation(sid, mid, lead, as_of_s),
            citation_long=citation_long(sid, st_name, mid, lead, data_through, as_of_s),
            canonical=f"{SITE_URL}{permalink_url(sid, mid, lead)}",
        )
        w.write(f"{base}/index.html", html)
        n += 1
    return n


def _write_permalink_csv(path: Path, sid: str, mid: str, lead: int, ser: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["station_id,model_id,lead_day,init_hour,method,variable,climo_date,error_f"]
    if ser:
        for d, v in zip(ser["dates"], ser["values"]):
            if v is None:
                continue
            lines.append(f"{sid},{mid},{lead},{DEFAULT_INIT},{DEFAULT_METHOD},"
                         f"{DEFAULT_VARIABLE},{d},{v:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------
# fixed pages, downloads, feed and headers
# ------------------------------------------------------------------------------------------

def _prune_variants(out: Path) -> None:
    """Delete the generated ``v/`` view directories so a renamed slug cannot survive a rebuild."""
    for path in (out / "v", *(out / "station").glob("*/v"), *(out / "model").glob("*/v")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


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
        if not f.is_file():
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


def _write_status(env, w, base_ctx, report, names: dict[str, str] | None = None) -> None:
    view = _status_view(report, names)
    w.write("status/index.html", env.get_template("status.html").render(
        **base_ctx, report=report, **view))


def _status_view(report: dict | None, names: dict[str, str] | None = None) -> dict:
    """Turn the raw completeness report into uptime bars and a headline state."""
    if not report:
        return {"model_bars": [], "truth_bars": [], "overall": "unknown",
                "overall_text": "No status report has been generated yet."}
    days = report.get("days", 0)

    def bar(rows, key_yes, key_part, label):
        flags = ["yes" if d.get(key_yes) else ("part" if d.get(key_part) else "no") for d in rows]
        pct = 100.0 * sum(1 for f in flags if f == "yes") / max(len(flags), 1)
        return {"svg": Markup(svg.availability_row(flags, label=label)),
                "uptime": f"{pct:.1f}%", "flags": flags}

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

    if report.get("ok"):
        overall, text = "ok", "All systems operational — nothing is missing for the current day."
    elif report.get("n_current_gaps", 0) > 0:
        overall = "bad"
        text = (f"{report['n_current_gaps']} item(s) missing for the current day; "
                f"{report.get('n_gaps', 0)} over the last {days} days.")
    else:  # pragma: no cover - defensive
        overall, text = "warn", "Degraded."
    return {"model_bars": model_bars, "truth_bars": truth_bars,
            "overall": overall, "overall_text": text}


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

    if scores is not None and len(scores):
        # 4 dp in °C is 0.0001 °C — far finer than anything measurable here, and it keeps the
        # published CSV a third smaller than float64 repr would.
        scores.round(4).to_csv(d / "scores_latest.csv", index=False)
    else:
        (d / "scores_latest.csv").write_text(",".join(SCORE_COLUMNS) + "\n", encoding="utf-8")
    add("scores_latest.csv", "every published aggregate: station × model × init × lead × variable "
        "× method × window, with n, MAE, bias, RMSE, hit rates, skill and bootstrap intervals "
        "(values in °C).", len(scores) if scores is not None else 0)

    with gzip.open(d / "pairwise_latest.csv.gz", "wt", encoding="utf-8", newline="") as f:
        if pairwise is not None and len(pairwise):
            pairwise.round(4).to_csv(f, index=False)
        else:
            f.write(",".join(PAIRWISE_COLUMNS) + "\n")
    add("pairwise_latest.csv.gz", "paired model-vs-model MAE differences on common days, with the "
        "bootstrap interval and the significance flag (°C).",
        len(pairwise) if pairwise is not None else 0)

    n_err = 0
    cols = ["station_id", "model_id", "init_hour", "lead_day", "method", "variable",
            "climo_date", "fcst_c", "obs_c", "err"]
    with gzip.open(d / "daily_errors.csv.gz", "wt", encoding="utf-8", newline="") as f:
        if errors is not None and len(errors):
            e = errors[[c for c in cols if c in errors.columns]].copy()
            e["climo_date"] = pd.to_datetime(e["climo_date"]).dt.date
            e.round(4).to_csv(f, index=False)
            n_err = len(e)
        else:
            f.write(",".join(cols) + "\n")
    add("daily_errors.csv.gz", "the complete per-day record behind every score: one row per "
        "station, model, initialization, lead day, method, variable and climatological day, with "
        "the forecast, the observation and the signed error (°C). This is the file to download if "
        "you want to recompute anything.", n_err)

    st_lines = ["station_id,name,cli_pil,tz,std_offset_h,lat,lon,elev_m,kalshi"]
    for s in stations:
        st_lines.append(",".join(str(x) if x is not None else "" for x in (
            s.id, f'"{s.name}"', s.cli_pil, s.tz, s.std_offset_h, s.lat, s.lon, s.elev_m,
            s.kalshi or "")))
    (d / "stations.csv").write_text("\n".join(st_lines) + "\n", encoding="utf-8")
    add("stations.csv", "station metadata: identifier, name, truth product, fixed standard UTC "
        "offset, coordinates and elevation.", len(stations))

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


def _write_data(env, w, base_ctx, api_written, scores, pairwise, downloads) -> None:
    api_written = api_written or {}
    endpoints = [
        ("scores/latest.json", "every published score (all stations, models, leads, windows)",
         api_written.get("scores/latest.json", len(scores))),
        ("scores/leaderboard.json", "the station_id=ALL slice used by the front page",
         api_written.get("scores/leaderboard.json", "—")),
        ("leaderboard/{window}-{init}z-{method}-{variable}.json",
         "one pre-built file per leaderboard view (32 of them)",
         api_written.get("leaderboard/{view}.json", "—")),
        ("scores/{station}/{model}/{lead}.json",
         "one permanent-link card plus the last 90 days of daily errors",
         api_written.get("scores/{station}/{model}/{lead}.json", "—")),
        ("pairwise/latest.json", "model-vs-model paired bootstrap (station_id=ALL)",
         api_written.get("pairwise/latest.json", len(pairwise))),
        ("stations.json", "station metadata", api_written.get("stations.json", "—")),
        ("models.json", "model metadata", api_written.get("models.json", "—")),
        ("status.json", "pipeline completeness report", api_written.get("status.json", "—")),
        ("openapi.json", "OpenAPI 3.1 description of the endpoints above",
         api_written.get("openapi.json", "—")),
    ]
    schemas = [
        {"name": "forecast_values", "what": "one extracted station value per model run, valid "
         "time, variable and interpolation method (DESIGN §3.1).",
         "columns": ["model_id", "model_version", "init_time", "valid_time", "lead_h",
                     "station_id", "variable", "bucket_h", "method", "value_c", "missing_reason",
                     "source_url", "fetched_at"]},
        {"name": "truth_daily", "what": "one row per station-day-source with the first-final "
         "policy and QC flags (DESIGN §3.2).", "columns": TRUTH_COLUMNS},
        {"name": "daily_forecasts", "what": "sampled and native daily extremes per model run, "
         "station and climatological day (DESIGN §3.3).", "columns": DAILY_COLUMNS},
        {"name": "scores", "what": "published aggregates with bootstrap intervals (DESIGN §3.4).",
         "columns": SCORE_COLUMNS},
        {"name": "pairwise", "what": "paired model-vs-model MAE differences (DESIGN §3.5).",
         "columns": PAIRWISE_COLUMNS},
    ]
    for t in schemas:
        t["fields"] = [
            {"name": c, "type": COLUMN_DOCS.get(c, ("—", "—", ""))[0],
             "unit": COLUMN_DOCS.get(c, ("—", "—", ""))[1],
             "what": COLUMN_DOCS.get(c, ("—", "—", ""))[2]}
            for c in t["columns"]
        ]
    w.write("data/index.html", env.get_template("data.html").render(
        **base_ctx,
        endpoints=[{"path": p, "href": f"/api/v1/{p}" if "{" not in p else "/api/v1/",
                    "what": what, "n": n} for p, what, n in endpoints],
        schemas=schemas,
        downloads=downloads,
        changelog=CHANGELOG,
        limitations=LIMITATIONS,
        data_citation=(
            f"CastCheck (2026). Station-level verification of raw weather-model 2 m temperature "
            f"forecasts [data set]. Methodology version {METHODOLOGY_VERSION}, schema version "
            f"{SCHEMA_VERSION}, data through {base_ctx['data_through']}. {SITE_URL}"),
    ))


def _write_indexes(env, w, base_ctx, scores, stations, model_idx) -> None:
    """``/stations/`` and ``/models/``: the two directories the top navigation points at.

    Both are built from the registry, so they exist and are complete even before a single day has
    been scored; the sample sizes come from the default view and are simply blank until then.
    """
    names = _display_map(model_idx)
    head = pd.DataFrame()
    if scores is not None and len(scores):
        sc = scores
        head = sc[(sc["window"] == DEFAULT_WINDOW) & (sc["init_hour"].astype(int) == DEFAULT_INIT)
                  & (sc["method"] == DEFAULT_METHOD) & (sc["variable"] == DEFAULT_VARIABLE)
                  & (sc["lead_day"].astype(int) == 1)]

    st_rows = []
    for st in stations:
        part = head[head["station_id"] == st.id] if len(head) else head
        n = int(part["n"].max()) if len(part) else 0
        st_rows.append({
            "id": st.id, "name": st.name, "cli_pil": st.cli_pil, "tz": st.tz,
            "offset": f"{st.std_offset_h:+d}",
            "elev": "—" if st.elev_m is None else f"{st.elev_m:.0f} m",
            "n": n or "—", "n_models": int(part["model_id"].nunique()) if len(part) else "—",
            "low_n": n < MIN_N,
        })

    md_rows = []
    for mid in sorted(model_idx, key=lambda k: (model_idx[k].get("baseline", False), names[k])):
        info = model_idx[mid]
        part = head[(head["model_id"] == mid) & (head["station_id"] == ALL_STATIONS)] \
            if len(head) else head
        allw = pd.DataFrame()
        if scores is not None and len(scores):
            allw = scores[(scores["model_id"] == mid) & (scores["window"] == "all")]
        n = int(part["n"].max()) if len(part) else 0
        md_rows.append({
            "model_id": mid, "name": names[mid], "source": info.get("source", ""),
            "product": info.get("product", ""), "init_field": info.get("init_field"),
            "baseline": info.get("baseline", False),
            "n": n or "—", "low_n": n < MIN_N,
            "period": f_period(allw["period_start"].min(), allw["period_end"].max())
            if len(allw) else "—",
            **_segment_note(allw),
        })

    shared = {"window_label": f"the last {DEFAULT_WINDOW[:-1]} days",
              "init_hour": f"{DEFAULT_INIT:02d}", "method": DEFAULT_METHOD}
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
        dates = sorted((p.stem for p in hist.glob("*.parquet")), reverse=True)[:30]
    if as_of_s not in dates:
        dates = [as_of_s, *dates]
    board = sc[(sc["station_id"] == ALL_STATIONS) & (sc["window"] == DEFAULT_WINDOW)
               & (sc["init_hour"] == DEFAULT_INIT) & (sc["method"] == DEFAULT_METHOD)
               & (sc["variable"] == DEFAULT_VARIABLE) & (sc["lead_day"] == 1)
               & (sc["model_id"] != PERSISTENCE_ID)]
    board = board[board["n"] >= MIN_N] if len(board) else board
    if len(board):
        top = board.sort_values("mae").iloc[0]
        summary = (f"Lead day 1, daily maximum, {DEFAULT_WINDOW} window, "
                   f"{DEFAULT_INIT:02d}Z, {DEFAULT_METHOD}: lowest MAE is "
                   f"{f_delta(top['mae'])} °F ({_mname(model_idx, top['model_id'])}, "
                   f"n = {int(top['n'])}).")
    else:
        summary = (f"No group in the default view has reached {MIN_N} scored days yet, so nothing "
                   f"is ranked; every number is published with its sample size.")
    return [{"date": d, "summary": summary if i == 0 else
             "Scores recomputed from scratch for this publication date.",
             # the newest entry is stamped with the actual build, older ones with their slot
             "updated": built_at if i == 0 else f"{d}T{PUBLISH_HOUR_UTC:02d}:00:00+00:00"}
            for i, d in enumerate(dates[:30])]


def _write_feed(env, w, base_ctx, entries) -> None:
    w.write("feed.xml", env.get_template("feed.xml").render(
        **base_ctx, entries=entries or []), count=False)
