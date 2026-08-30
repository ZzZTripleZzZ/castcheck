"""Static site generator (DESIGN §6).

Jinja2 → ``public/``.  Every page is readable with JavaScript disabled: the tables are the content
and the one small chart script is an enhancement that fails silently.  Routes:

``/``                                          leaderboard (all stations, 90 d, 00Z, bilinear)
``/v/{init}z-{method}/``                       the same leaderboard for the other three combinations
``/station/{ICAO}/`` (+ ``/v/…/``)             one station, all models × leads
``/model/{model_id}/`` (+ ``/v/…/``)           one model, all stations × leads
``/station/{ICAO}/model/{model_id}/lead/{d}/`` the permanent link: every window/init/method for that
                                               combination, its pairwise comparisons, the daily error
                                               series and a citation block
``/methodology/``  ``/status/``  ``/data/``

``station_id="ALL"`` is published as a pseudo-station so the aggregate has permanent links too.
Errors are stored in °C (METHODOLOGY §3) and displayed in °F.
"""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .. import METHODOLOGY_VERSION, SCHEMA_VERSION, __version__
from ..config import PUBLIC_DIR, REPO_ROOT, ModelSpec, Station, load_models, load_stations
from ..store import DAILY_COLUMNS, TRUTH_COLUMNS
from ..verify import (
    ALL_STATIONS,
    MIN_N,
    PAIRWISE_COLUMNS,
    PERSISTENCE_ID,
    SCORE_COLUMNS,
    select_truth,
)

__all__ = ["FAIRNESS", "SITE_URL", "build_site", "citation"]

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
ASSETS = HERE / "assets"

SITE_URL = "https://castcheck.zifanzhang.com"

#: METHODOLOGY §7, shown at the top of every page.
FAIRNESS = (
    "These are raw model outputs on the native 0.25° grid, without MOS, bias correction, "
    "downscaling or any post-processing. They are not equivalent to the products end users receive "
    "from a weather service or app, and the scores here understate operational forecast quality."
)

HEADLINE_LEADS = (1, 3, 5, 7)
DEFAULT_WINDOW = "90d"
DEFAULT_INIT = 0
DEFAULT_METHOD = "bilinear"
DEFAULT_VARIABLE = "tmax"
VARIANTS = ((0, "bilinear"), (0, "nearest"), (12, "bilinear"), (12, "nearest"))
C_TO_F_DELTA = 9.0 / 5.0
RECENT_TRUTH_DAYS = 30
WINDOW_ORDER = {"30d": 0, "90d": 1, "365d": 2, "all": 3}


# ------------------------------------------------------------------------------------------
# formatting helpers
# ------------------------------------------------------------------------------------------

def _isnan(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def f_delta(c, digits: int = 2) -> str:
    """A temperature *difference* in °C rendered in °F."""
    return "—" if _isnan(c) else f"{float(c) * C_TO_F_DELTA:.{digits}f}"


def f_signed(c, digits: int = 2) -> str:
    return "—" if _isnan(c) else f"{float(c) * C_TO_F_DELTA:+.{digits}f}"


def f_ci(lo, hi, digits: int = 2) -> str:
    if _isnan(lo) or _isnan(hi):
        return "—"
    return f"{float(lo) * C_TO_F_DELTA:.{digits}f} – {float(hi) * C_TO_F_DELTA:.{digits}f}"


def f_pct(x) -> str:
    return "—" if _isnan(x) else f"{float(x) * 100:.0f} %"


def f_skill(x) -> str:
    return "—" if _isnan(x) else f"{float(x):+.2f}"


def f_period(start, end) -> str:
    if _isnan(start) or _isnan(end):
        return "—"
    return f"{start} → {end}"


def permalink_url(station_id: str, model_id: str, lead: int) -> str:
    return f"/station/{station_id}/model/{model_id}/lead/{int(lead)}/"


def citation(station_id: str, model_id: str, lead: int, accessed: str) -> str:
    return (
        f"CastCheck, {station_id} {model_id} lead {int(lead)}, "
        f"methodology v{METHODOLOGY_VERSION}, accessed {accessed}, "
        f"{SITE_URL}{permalink_url(station_id, model_id, lead)}"
    )


def _variant_href(base: str, init_hour: int, method: str) -> str:
    if (int(init_hour), method) == (DEFAULT_INIT, DEFAULT_METHOD):
        return base
    return f"{base}v/{int(init_hour):02d}z-{method}/"


def _variants(base: str, init_hour: int, method: str) -> list[dict]:
    return [
        {
            "href": _variant_href(base, ih, m),
            "label": f"{ih:02d}Z · {m}",
            "current": (int(ih), m) == (int(init_hour), method),
        }
        for ih, m in VARIANTS
    ]


# ------------------------------------------------------------------------------------------
# generator
# ------------------------------------------------------------------------------------------

def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


class _Writer:
    def __init__(self, out: Path):
        self.out = out
        self.n = 0

    def write(self, relpath: str, html: str) -> Path:
        path = self.out / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        self.n += 1
        return path


def _render_markdown(text: str) -> Markup:
    try:
        import markdown as md

        return Markup(md.markdown(text, extensions=["tables", "toc", "attr_list"]))
    except Exception:  # noqa: BLE001  # pragma: no cover - markdown is a soft dependency
        from markupsafe import escape

        return Markup(f"<pre class='mono'>{escape(text)}</pre>")


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


def _row_view(r, station_id: str, model_id: str, lead: int) -> dict:
    return {
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
        "mae_ci": f_ci(r["mae_ci_low"], r["mae_ci_high"]),
        "bias": f_signed(r["bias"]),
        "bias_ci": f_ci(r["bias_ci_low"], r["bias_ci_high"]),
        "rmse": f_delta(r["rmse"]),
        "hit1f": f_pct(r["hit1f"]),
        "hit2f": f_pct(r["hit2f"]),
        "hit3f": f_pct(r["hit3f"]),
        "skill": f_skill(r["skill_persistence"]),
        "period": f_period(r["period_start"], r["period_end"]),
        "permalink": permalink_url(station_id, model_id, lead),
    }


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
    base_ctx = {
        "fairness": FAIRNESS,
        "methodology_version": METHODOLOGY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "castcheck_version": __version__,
        "as_of": as_of_s,
        "built_at": built_at,
        "min_n": MIN_N,
    }

    # assets
    asset_dir = out / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(ASSETS.glob("*")):
        if f.is_file():
            shutil.copyfile(f, asset_dir / f.name)

    counts = {"pages": 0, "stations": 0, "models": 0, "permalinks": 0, "leaderboards": 0}

    # ---- always-present pages -------------------------------------------------------------
    _write_methodology(env, w, base_ctx)
    _write_status(env, w, base_ctx, status_report)
    _write_data(env, w, base_ctx, api_written, scores, pairwise)

    if scores is None or len(scores) == 0:
        w.write("index.html", env.get_template("empty.html").render(
            **base_ctx,
            heading="CastCheck — no scores yet",
            message="No forecast has been matched to an observation yet, so there is nothing to "
                    "rank. This page will fill in automatically once the first climatological day "
                    "has both a model forecast and its NWS Daily Climate Report.",
        ))
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
    station_idx = {s.id: s for s in stations}
    leads = sorted(sc["lead_day"].unique().tolist())
    station_links = [
        {"id": s.id, "name": s.name} for s in stations if s.id in set(sc["station_id"])
    ]

    # ---- leaderboards ---------------------------------------------------------------------
    for ih, meth in VARIANTS:
        html = _leaderboard_html(env, base_ctx, sc, model_idx, station_links, ih, meth, len(stations))
        rel = _variant_href("/", ih, meth).strip("/")
        w.write(f"{rel}/index.html" if rel else "index.html", html)
        counts["leaderboards"] += 1

    # ---- station pages --------------------------------------------------------------------
    truth_sel = select_truth(truth) if truth is not None and len(truth) else pd.DataFrame()
    all_station = Station(id=ALL_STATIONS, name="All stations (mean of daily station errors)",
                          cli_pil="", tz="UTC", std_offset_h=0, lat=None, lon=None, elev_m=None)
    for sid in sorted(sc["station_id"].unique().tolist()):
        st = station_idx.get(sid, all_station if sid == ALL_STATIONS else None)
        if st is None:
            st = Station(id=sid, name=sid, cli_pil="", tz="UTC", std_offset_h=0,
                         lat=None, lon=None, elev_m=None)
        sub = sc[sc["station_id"] == sid]
        recent = _recent_truth(truth_sel, sid)
        for ih, meth in VARIANTS:
            html = _grid_html(
                env, "station.html", base_ctx, sub, ih, meth, leads,
                row_key="model_id", row_label="model_id", station_fixed=sid,
                extra={"station": st, "recent": recent,
                       "variants": _variants(f"/station/{sid}/", ih, meth)},
                model_idx=model_idx,
            )
            rel = _variant_href(f"/station/{sid}/", ih, meth).strip("/")
            w.write(f"{rel}/index.html", html)
        counts["stations"] += 1

    # ---- model pages ----------------------------------------------------------------------
    for mid in sorted(sc["model_id"].unique().tolist()):
        sub = sc[(sc["model_id"] == mid) & (sc["station_id"] != ALL_STATIONS)]
        if sub.empty:
            sub = sc[sc["model_id"] == mid]
        info = model_idx.get(mid, {"model_id": mid, "family": mid, "source": "?", "product": "?",
                                   "init_field": None, "inits": [0, 12], "step_h": 6, "max_h": 240,
                                   "native_extremes": [], "baseline": False})
        for ih, meth in VARIANTS:
            html = _grid_html(
                env, "model.html", base_ctx, sub, ih, meth, leads,
                row_key="station_id", row_label="station_id", station_fixed=None, model_fixed=mid,
                extra={"model": info, "variants": _variants(f"/model/{mid}/", ih, meth)},
                model_idx=model_idx,
            )
            rel = _variant_href(f"/model/{mid}/", ih, meth).strip("/")
            w.write(f"{rel}/index.html", html)
        counts["models"] += 1

    # ---- permanent links ------------------------------------------------------------------
    if permalinks:
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
        for (sid, mid, lead), grp in sc.groupby(["station_id", "model_id", "lead_day"], observed=True):
            lead = int(lead)
            grp = grp.assign(_w=grp["window"].map(WINDOW_ORDER).fillna(9))
            rows = [_row_view(r, sid, mid, lead) for _, r in
                    grp.sort_values(["variable", "init_hour", "method", "_w"]).iterrows()]
            pairs = _pair_views(pw_idx.get((sid, lead)), mid)
            st = station_idx.get(sid)
            html = tpl.render(
                **base_ctx,
                station_id=sid, model_id=mid, lead=lead,
                station_name=st.name if st else ("All stations" if sid == ALL_STATIONS else sid),
                model_family=model_idx.get(mid, {}).get("family", mid),
                rows=rows, pairwise=pairs, pairwise_window=DEFAULT_WINDOW,
                chart_init=f"{DEFAULT_INIT:02d}", chart_method=DEFAULT_METHOD,
                chart_variable=DEFAULT_VARIABLE,
                series_days=90,
                json_url=f"/api/v1/scores/{sid}/{mid}/{lead}.json",
                citation=citation(sid, mid, lead, as_of_s),
            )
            w.write(f"station/{sid}/model/{mid}/lead/{lead}/index.html", html)
            counts["permalinks"] += 1

    counts["pages"] = w.n
    return counts


# ------------------------------------------------------------------------------------------
# page builders
# ------------------------------------------------------------------------------------------

def _leaderboard_html(env, base_ctx, sc, model_idx, station_links, init_hour, method, n_stations):
    sub = sc[
        (sc["station_id"] == ALL_STATIONS)
        & (sc["window"] == DEFAULT_WINDOW)
        & (sc["init_hour"] == int(init_hour))
        & (sc["method"] == method)
    ]
    boards = []
    for lead in HEADLINE_LEADS:
        for var in ("tmax", "tmin"):
            part = sub[(sub["lead_day"] == lead) & (sub["variable"] == var)]
            if part.empty:
                continue
            part = part.sort_values("mae")
            rows = []
            for _, r in part.iterrows():
                v = _row_view(r, ALL_STATIONS, r["model_id"], lead)
                v["baseline"] = model_idx.get(r["model_id"], {}).get("baseline", False)
                rows.append(v)
            ranked = [r for r in rows if not (r["low_n"] or r["baseline"])]
            others = [r for r in rows if r["low_n"] or r["baseline"]]
            boards.append({"lead": lead, "variable": var, "rows": ranked + others})

    avail = sc[
        (sc["station_id"] == ALL_STATIONS) & (sc["window"] == "all")
        & (sc["variable"] == DEFAULT_VARIABLE) & (sc["init_hour"] == int(init_hour))
        & (sc["method"] == method) & (sc["lead_day"] == 1)
    ]
    availability, avail_start, avail_end = _availability(avail)

    return env.get_template("index.html").render(
        **base_ctx,
        boards=boards,
        window_label=f"last {DEFAULT_WINDOW.rstrip('d')} days",
        init_hour=f"{int(init_hour):02d}",
        method=method,
        n_stations=n_stations,
        variants=_variants("/", init_hour, method),
        availability=availability,
        avail_start=avail_start,
        avail_end=avail_end,
        station_links=station_links,
    )


def _availability(avail: pd.DataFrame):
    if avail.empty:
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
            "model_id": r["model_id"], "n": int(r["n"]),
            "period": f_period(r["period_start"], r["period_end"]),
            "left": round(left, 2), "width": round(min(width, 100.0 - left), 2),
        })
    return out, lo.date().isoformat(), hi.date().isoformat()


def _grid_html(env, template, base_ctx, sub, init_hour, method, leads, *, row_key, row_label,
               station_fixed=None, model_fixed=None, extra=None, model_idx=None):
    part = sub[
        (sub["window"] == DEFAULT_WINDOW)
        & (sub["init_hour"] == int(init_hour))
        & (sub["method"] == method)
    ]
    blocks = []
    for var in ("tmax", "tmin"):
        p = part[part["variable"] == var]
        if p.empty:
            continue
        rows = []
        for key in sorted(p[row_key].unique().tolist()):
            q = p[p[row_key] == key].set_index("lead_day")
            cells = []
            for lead in leads:
                if lead in q.index:
                    r = q.loc[lead]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    sid = station_fixed if station_fixed is not None else key
                    mid = model_fixed if model_fixed is not None else key
                    cells.append({
                        "mae": f_delta(r["mae"]), "n": int(r["n"]),
                        "low_n": int(r["n"]) < MIN_N,
                        "permalink": permalink_url(sid, mid, lead),
                    })
                else:
                    cells.append({"mae": "", "n": 0, "low_n": True, "permalink": ""})
            rows.append({
                row_label: key, "cells": cells,
                "baseline": bool(model_idx.get(key, {}).get("baseline", False)) if model_idx and row_key == "model_id" else False,
            })
        blocks.append({"variable": var, "rows": rows})
    ctx = dict(base_ctx)
    ctx.update(extra or {})
    return env.get_template(template).render(
        **ctx, blocks=blocks, leads=leads,
        window_label=f"last {DEFAULT_WINDOW.rstrip('d')} days",
        init_hour=f"{int(init_hour):02d}", method=method,
    )


def _pair_views(grp: pd.DataFrame | None, model_id: str) -> list[dict]:
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


def _write_methodology(env, w, base_ctx) -> None:
    path = REPO_ROOT / "METHODOLOGY.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Methodology\n\nNot available."
    w.write("methodology/index.html", env.get_template("page.html").render(
        **base_ctx, heading="Methodology", body=_render_markdown(text)))


def _write_status(env, w, base_ctx, report) -> None:
    w.write("status/index.html", env.get_template("status.html").render(**base_ctx, report=report))


def _write_data(env, w, base_ctx, api_written, scores, pairwise) -> None:
    api_written = api_written or {}
    endpoints = [
        ("scores/latest.json", "every published score (all stations, models, leads, windows)",
         api_written.get("scores/latest.json", len(scores))),
        ("scores/leaderboard.json", "the station_id=ALL slice used by the front page",
         api_written.get("scores/leaderboard.json", "—")),
        ("scores/{station}/{model}/{lead}.json",
         "one permanent-link card plus the last 90 days of daily errors",
         api_written.get("scores/{station}/{model}/{lead}.json", "—")),
        ("pairwise/latest.json", "model-vs-model paired bootstrap (station_id=ALL)",
         api_written.get("pairwise/latest.json", len(pairwise))),
        ("stations.json", "station metadata", api_written.get("stations.json", "—")),
        ("models.json", "model metadata", api_written.get("models.json", "—")),
        ("status.json", "pipeline completeness report", api_written.get("status.json", "—")),
    ]
    schemas = [
        {"name": "forecast_values", "what": "one extracted station value per model run, valid time, "
         "variable and interpolation method (DESIGN §3.1).",
         "columns": ["model_id", "model_version", "init_time", "valid_time", "lead_h", "station_id",
                     "variable", "bucket_h", "method", "value_c", "missing_reason", "source_url",
                     "fetched_at"]},
        {"name": "truth_daily", "what": "one row per station-day-source with the first-final policy "
         "and QC flags (DESIGN §3.2).", "columns": TRUTH_COLUMNS},
        {"name": "daily_forecasts", "what": "sampled and native daily extremes per model run, "
         "station and climatological day (DESIGN §3.3).", "columns": DAILY_COLUMNS},
        {"name": "scores", "what": "published aggregates with bootstrap intervals (DESIGN §3.4).",
         "columns": SCORE_COLUMNS},
        {"name": "pairwise", "what": "paired model-vs-model MAE differences (DESIGN §3.5).",
         "columns": PAIRWISE_COLUMNS},
    ]
    w.write("data/index.html", env.get_template("data.html").render(
        **base_ctx,
        endpoints=[{"path": p, "href": f"/api/v1/{p}" if "{" not in p else "/api/v1/scores/",
                    "what": what, "n": n} for p, what, n in endpoints],
        schemas=schemas,
    ))
