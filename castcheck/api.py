"""Static JSON API export (DESIGN §6).

Everything under ``public/api/v1`` is a plain file; there is no server.  Large tables use a compact
``{"columns": [...], "rows": [[...]]}`` encoding, constant columns (versions, ``computed_at``) are
lifted into the envelope, and floats are rounded so the payload stays small enough to be served
from a CDN and read by the ~4 KB chart script.

Endpoints
---------
``stations.json``                              station metadata
``models.json``                                model metadata (incl. the persistence baseline)
``scores/index.json``                          the index of the sharded scores export: which
                                               stations, models, leads and views exist, and the
                                               path of every shard.  ``scores/latest.json`` serves
                                               the same document (the whole table was a single
                                               21 MB file, past what a CDN edge will serve well and
                                               close to the 25 MiB Cloudflare Pages limit)
``scores/by-station/{station}.json``           a pointer at the station's bundle (kept so an
                                               existing link resolves)
``scores/leaderboard.json``                    the ``station_id="ALL"`` slice only (small; used by ``/``)
``/station/{station}/cards.json``              **the per-station bundle**: every score for that
                                               station under ``scores``, and one entry in
                                               ``cards`` per model and lead day carrying that
                                               card's pairwise comparisons and the last 90 days of
                                               signed daily error.  A permanent-link page links to
                                               its own card by fragment (``…/cards.json#gfs-1``).
                                               The matching forecasts and observations are in
                                               ``/data/daily_errors/{station}.csv.gz``
``pairwise/latest.json``                       the ``station_id="ALL"`` pairwise slice
``leaderboard/{window}-{init}z-{method}-{variable}.json``
                                               one pre-ranked file per site view, results as
                                               objects with a ``permalink`` each
``openapi.json``                               OpenAPI 3.1 description of all of the above
``status.json``                                data-completeness report (see ``status.py``)
``diagnostics.json``                           the diurnal-structure diagnostic behind
                                               ``/diagnostics/``: per-model bias at each of the
                                               four synoptic instants, bias against lead day, and
                                               the sampling penalty of the ``*_cli`` variables

Every response carries the same envelope (docs/05 §D): ``schema_version``, ``generated_at``,
``data_through``, ``next_update``, ``window {type, days, start, end}``, ``units``,
``method {ci, resamples, level, block, ref}`` and ``truth {source}``.  Every leaderboard row
carries a literal ``permalink`` to the page where the number is explained; the bulk per-station
table carries ``model_id`` and ``lead_day`` and the envelope's ``permalink_template``, which is
the same information without forty bytes on each of a quarter of a million rows.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION, __version__
from .config import PUBLIC_DIR, ModelSpec, Station, load_models, load_stations
from .verify import ALL_STATIONS, BLOCK_DAYS, MIN_N, PERSISTENCE_ID, error_table

__all__ = [
    "DIAGNOSTICS_HOURS",
    "DIAGNOSTICS_INIT",
    "DIAGNOSTICS_LEAD",
    "DIAGNOSTICS_LEADS",
    "DIAGNOSTICS_METHOD",
    "DIAGNOSTICS_PENALTY_PAIRS",
    "DIAGNOSTICS_WINDOW",
    "LEADERBOARD_VIEWS",
    "SERIES_DAYS",
    "SKILL_MIN_COMMON",
    "UNITS",
    "api_dir",
    "compact_table",
    "diagnostics_payload",
    "export_api",
    "openapi_document",
    "write_json",
]

SERIES_DAYS = 90
#: A skill score divides two MAEs computed on the *common* days of the model and the baseline.
#: Below this many common days the ratio is dominated by which days happened to intersect, so the
#: site prints "—" for it and every published row carries ``skill_reliable`` saying which side of
#: the threshold it is on.  The number itself is never withheld from the JSON.
SKILL_MIN_COMMON = 10
#: The card's daily series is published for the headline slice only (see :func:`_daily_series`).
SERIES_INIT = 0
SERIES_METHOD = "bilinear"
#: Window of the pairwise block inside a permanent-link card.
CARD_PAIRWISE_WINDOW = "90d"
N_BOOT = 1000
CI_LEVEL = 0.95
_ENVELOPE_CONSTANTS = ("computed_at", "methodology_version", "schema_version")

#: Units of every numeric field, so a consumer never has to guess.  Errors are computed and
#: published in °C; the HTML site converts differences to °F for display.
UNITS = {
    "mae": "degC", "bias": "degC", "rmse": "degC",
    "mae_ci_low": "degC", "mae_ci_high": "degC",
    "bias_ci_low": "degC", "bias_ci_high": "degC",
    "mae_diff": "degC", "ci_low": "degC", "ci_high": "degC",
    "n": "days", "n_common": "days", "n_debiased": "days",
    "hit1f": "fraction", "hit2f": "fraction", "hit3f": "fraction",
    "hit1f_ci_low": "fraction", "hit1f_ci_high": "fraction",
    "skill_persistence": "fraction", "skill_persistence_debiased": "fraction",
    "skill_ci_low": "fraction", "skill_ci_high": "fraction",
    "mae_persistence_common": "degC", "mae_debiased": "degC",
    "p_boot": "probability",
    "init_hour": "UTC hour", "lead_day": "days",
}

METHOD_BLOCK = {
    "ci": "circular moving-block bootstrap on each group's own realized date axis",
    "resamples": N_BOOT,
    "level": CI_LEVEL,
    "block": f"{BLOCK_DAYS} days",
    "min_n": MIN_N,
    "ci_min_n": 28,
    "ci_min_blocks": 4,
    "ci_absent": "null when n < ci_min_n or fewer than ci_min_blocks blocks",
    "proportions": "Wilson score intervals (hit rates), not bootstrapped",
    "skill_denominator": "mae_persistence_common on n_common days",
    "debiasing": "out-of-sample: bias of the trailing 30 scored days before each day, min 15",
    "multiplicity": "pairwise carries distinguishable_uncorrected, p_boot and "
                    "distinguishable_holm; only the Holm flag is marked on the site",
    "skill_min_n_common": SKILL_MIN_COMMON,
    "skill_reliable": f"false when n_common < {SKILL_MIN_COMMON}; the value is still published, "
                      f"but the site shows an em dash for it",
    "ref": "https://castcheck.zifanzhang.com/methodology/",
}

TRUTH_BLOCK = {
    "source": "observed 2 m temperature at 00/06/12/18 UTC (routine METAR within +/-35 min) for "
              "t2* and for tmax_s/tmin_s; NWS Daily Climate Report (CLI), first final issuance "
              "after local midnight, for tmax_cli/tmin_cli",
    "fallback": ["CF6 monthly summary", "hourly station observations (flagged)"],
    "policy": "first-final; later corrections are stored but never change a published score",
}

WINDOWS = ("30d", "90d", "365d", "all")
INITS = (0, 12)
METHODS = ("bilinear", "nearest")
#: The variables that get their own leaderboard file: the instantaneous headline and the two
#: like-for-like sampled extremes.  The secondary ``*_cli`` variables are in the score shards.
VARIABLES = ("t2", "tmax_s", "tmin_s")
#: every pre-built leaderboard file, mirroring the site's ``/v/…/`` pages
LEADERBOARD_VIEWS = tuple(
    (w, i, m, v) for w in WINDOWS for i in INITS for m in METHODS for v in VARIABLES
)
SITE = "https://castcheck.zifanzhang.com"
PUBLISH_HOUR_UTC = 11

#: The one slice ``/diagnostics/`` and ``diagnostics.json`` are cut from.  Fixing every dimension
#: but the one under examination is the whole point of a diagnostic: the diurnal figure varies the
#: valid instant, the lead figure varies the lead day, and the sampling-penalty table varies the
#: truth definition — nothing else moves between them.
DIAGNOSTICS_WINDOW = "90d"
DIAGNOSTICS_INIT = 0
DIAGNOSTICS_METHOD = "bilinear"
DIAGNOSTICS_LEAD = 1
#: The four common synoptic instants, in UTC.  At the 23 U.S. stations 18Z is early-to-late
#: afternoon local time and 12Z is around dawn; the page says so in words, because "18Z" is not
#: a time of day to a reader who does not already live in UTC.
DIAGNOSTICS_HOURS = (0, 6, 12, 18)
DIAGNOSTICS_LEADS = tuple(range(1, 10))
#: The like-for-like extreme and the NWS-daily-extreme comparison of the same forecast samples.
#: Their difference is the sampling penalty (METHODOLOGY §2.3), one number per model.
DIAGNOSTICS_PENALTY_PAIRS = (("tmax_s", "tmax_cli"), ("tmin_s", "tmin_cli"))


def api_dir(out: str | Path | None = None) -> Path:
    return Path(out) if out is not None else PUBLIC_DIR / "api" / "v1"


def _clean(value):
    """JSON-safe scalar: NaN/NaT → None, numpy scalars → python, dates → ISO strings."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        # 3 dp in °C is 0.001 °C — two orders of magnitude finer than the whole-°F truth.
        return None if math.isnan(v) else round(v, 3)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, (str, int)) else str(value)


def compact_table(df: pd.DataFrame, drop: tuple[str, ...] = _ENVELOPE_CONSTANTS) -> dict:
    """``{"columns": [...], "rows": [[...]]}`` with constant metadata columns removed."""
    if df is None or len(df) == 0:
        return {"columns": [], "rows": []}
    keep = [c for c in df.columns if c not in drop]
    sub = df[keep]
    rows = [[_clean(v) for v in rec] for rec in sub.itertuples(index=False, name=None)]
    return {"columns": keep, "rows": rows}


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    tmp.replace(path)
    return path


def _period(scores: pd.DataFrame, window: str | None = None) -> tuple[str | None, str | None]:
    """First and last climatological day covered by ``scores`` (optionally one window)."""
    if scores is None or len(scores) == 0 or "period_end" not in scores:
        return None, None
    sub = scores if window is None else scores[scores["window"] == window]
    if len(sub) == 0:
        return None, None
    start = pd.to_datetime(sub["period_start"], errors="coerce").dropna()
    end = pd.to_datetime(sub["period_end"], errors="coerce").dropna()
    return (start.min().date().isoformat() if len(start) else None,
            end.max().date().isoformat() if len(end) else None)


def _next_update(from_iso: str) -> str:
    """Next scheduled publish: 11:00 UTC daily (DESIGN §7)."""
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


def _window_block(scores: pd.DataFrame, window: str | None) -> dict:
    days = {"30d": 30, "90d": 90, "365d": 365, "all": None}
    start, end = _period(scores, window)
    return {
        "type": window or "multiple",
        "days": days.get(window) if window else None,
        "start": start,
        "end": end,
    }


def _envelope(scores: pd.DataFrame, window: str | None = None, **extra) -> dict:
    """The response envelope every endpoint shares (docs/05 §D)."""
    computed_at = ""
    if scores is not None and len(scores) and "computed_at" in scores:
        computed_at = str(scores["computed_at"].iloc[0])
    generated_at = computed_at or pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    _, data_through = _period(scores)
    env = {
        "castcheck_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "generated_at": generated_at,
        "computed_at": generated_at,  # kept for v1 compatibility; same value
        "data_through": data_through,
        "next_update": _next_update(generated_at),
        "window": _window_block(scores, window),
        "units": dict(UNITS),
        "method": dict(METHOD_BLOCK),
        "truth": dict(TRUTH_BLOCK),
        "license": "CC-BY-4.0 (CastCheck derived data); see /data/ for upstream licences",
        "site": SITE,
    }
    env.update(extra)
    return env


def permalink(station_id: str, model_id: str, lead_day: int) -> str:
    return f"/station/{station_id}/model/{model_id}/lead/{int(lead_day)}/"


def _with_permalink(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``permalink`` column required by docs/05 §D to a score-shaped table."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out["permalink"] = [
        permalink(s, m, ld)
        for s, m, ld in zip(out["station_id"], out["model_id"], out["lead_day"])
    ]
    return out


def _with_skill_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``skill_reliable``: was ``skill_persistence`` computed on enough common days?

    The score itself is never removed — a consumer who wants the ratio on four days can have it —
    but every row says which side of :data:`SKILL_MIN_COMMON` it falls on, so a table built from
    this file can grey the same cells the site greys.
    """
    if df is None or len(df) == 0 or "skill_reliable" in getattr(df, "columns", ()):
        return df
    out = df.copy()
    n_common = pd.to_numeric(out.get("n_common"), errors="coerce")
    out["skill_reliable"] = (n_common.fillna(-1) >= SKILL_MIN_COMMON) & out[
        "skill_persistence"].notna() if "skill_persistence" in out else (
        n_common.fillna(-1) >= SKILL_MIN_COMMON)
    return out


def _station_payload(stations: list[Station]) -> list[dict]:
    out = [{
        "id": s.id, "name": s.name, "cli_pil": s.cli_pil,
        "iem_id": getattr(s, "iem_id", None), "tz": s.tz,
        "std_offset_h": s.std_offset_h, "lat": s.lat, "lon": s.lon, "elev_m": s.elev_m,
        # DESIGN §10.4: the grid cell's own elevation and the station's height above it, so a
        # representativeness error can be told apart from a forecast error.
        "grid_elev_m": getattr(s, "grid_elev_m", None),
        "dz_m": getattr(s, "dz_m", None),
        "market_city": getattr(s, "market_city", None),
    } for s in stations]
    out.append({"id": ALL_STATIONS, "name": "All stations (mean of daily station errors)",
                "cli_pil": None, "iem_id": None, "tz": None, "std_offset_h": None,
                "lat": None, "lon": None, "elev_m": None, "grid_elev_m": None, "dz_m": None,
                "market_city": None})
    return out


def _model_payload(models: list[ModelSpec]) -> list[dict]:
    out = [{
        "model_id": m.model_id, "family": m.family, "source": m.source, "product": m.product,
        "init_field": m.init_field, "inits": list(m.inits), "step_h": m.step_h, "max_h": m.max_h,
        "native_extremes": list(m.native_extremes), "baseline": False,
    } for m in models]
    out.append({"model_id": PERSISTENCE_ID, "family": "Persistence (baseline)", "source": "truth",
                "product": "obs", "init_field": None, "inits": [0, 12], "step_h": 24,
                "max_h": 240, "native_extremes": [], "baseline": True})
    return out


def _daily_series(errors: pd.DataFrame, series_days: int) -> dict[tuple, list[dict]]:
    """(station_id, model_id, lead_day) → per-variable error series for the headline slice.

    Only the headline initialization and interpolation (00Z, bilinear) are carried here, for both
    variables. The card would otherwise repeat eight near-identical series per model, lead day and
    station; the other slices, and the matching forecasts and observations, are in
    ``/data/daily_errors/{station}.csv.gz``, which is 24 files instead of thousands.
    """
    if errors is None or len(errors) == 0:
        return {}
    e = errors[(errors["init_hour"].astype(int) == SERIES_INIT)
               & (errors["method"] == SERIES_METHOD)].copy()
    if len(e) == 0:
        return {}
    e["climo_date"] = pd.to_datetime(e["climo_date"])
    cutoff = e["climo_date"].max() - pd.Timedelta(days=series_days - 1)
    e = e[e["climo_date"] >= cutoff]
    out: dict[tuple, list[dict]] = {}
    keys = ["station_id", "model_id", "lead_day", "init_hour", "method", "variable"]
    # The pooled t2 variable carries four rows per day (one per common instant); a *series* needs
    # one point per calendar day, so those four are averaged. The per-instant values are in
    # /data/daily_errors/{station}.csv.gz.
    e = (e.groupby([*keys, "climo_date"], observed=True, as_index=False)["err"].mean())
    for key, grp in e.sort_values("climo_date").groupby(keys, observed=True):
        st, mid, lead, init_hour, method, variable = key
        # Only the signed error: this is a score card, and repeating the forecast and the
        # observation here would double every card for numbers that already live, in full and for
        # every day of the record, in /data/daily_errors/{station}.csv.gz.
        out.setdefault((st, mid, int(lead)), []).append({
            "init_hour": int(init_hour), "method": method, "variable": variable,
            "dates": [d.date().isoformat() for d in grp["climo_date"]],
            "err_c": [_clean(v) for v in grp["err"]],
        })
    return out


# ------------------------------------------------------------------------------------------
# diurnal-structure diagnostic (METHODOLOGY §10.2)
# ------------------------------------------------------------------------------------------

def _diag_index(scores: pd.DataFrame | None) -> dict[tuple[str, str, int], dict]:
    """``(variable, model_id, lead_day) -> row`` over the fixed diagnostic slice.

    Everything the diagnostic reads is already a published aggregate, so the page cannot drift
    from the leaderboard: it is the same ``scores`` table, cut along a different axis.
    """
    needed = {"station_id", "model_id", "variable", "lead_day", "window", "init_hour", "method"}
    if scores is None or len(scores) == 0 or not needed <= set(scores.columns):
        return {}
    sub = scores[
        (scores["station_id"] == ALL_STATIONS)
        & (scores["window"] == DIAGNOSTICS_WINDOW)
        & (scores["init_hour"].astype(int) == DIAGNOSTICS_INIT)
        & (scores["method"] == DIAGNOSTICS_METHOD)
    ]
    keep = [c for c in ("variable", "model_id", "lead_day", "n", "mae", "bias",
                        "bias_ci_low", "bias_ci_high", "period_start", "period_end")
            if c in sub.columns]
    return {
        (str(r["variable"]), str(r["model_id"]), int(r["lead_day"])): r
        for r in sub[keep].to_dict("records")
    }


def _diag_cell(row: dict | None, field: str = "bias"):
    return None if row is None else _clean(row.get(field))


def diagnostics_payload(scores: pd.DataFrame | None,
                        models: list[ModelSpec] | None = None) -> dict:
    """The three diagnostic cuts behind ``/diagnostics/``, in °C like every other endpoint.

    ``hourly_bias`` is the bias of each model at each of the four synoptic instants
    (``t2_00z … t2_18z``) at one lead day; ``bias_by_lead`` is the pooled ``t2`` bias against lead
    day; ``sampling_penalty`` is ``bias(*_cli) − bias(*_s)`` — how much of a model's daily-extreme
    bias is the four-instant sampling rather than the model.  No causal claim is attached to any
    of them here or on the page: see METHODOLOGY §10.2.
    """
    models = list(models) if models is not None else load_models()
    order = [m.model_id for m in models]
    idx = _diag_index(scores)
    present = {mid for _, mid, _ in idx}
    order = [m for m in order if m in present] + sorted(present - set(order))

    hourly = [
        {"model_id": mid,
         "bias": [_diag_cell(idx.get((f"t2_{h:02d}z", mid, DIAGNOSTICS_LEAD))) for h in
                  DIAGNOSTICS_HOURS],
         "n": [_diag_cell(idx.get((f"t2_{h:02d}z", mid, DIAGNOSTICS_LEAD)), "n") for h in
               DIAGNOSTICS_HOURS]}
        for mid in order
    ]
    by_lead = [
        {"model_id": mid,
         "bias": [_diag_cell(idx.get(("t2", mid, d))) for d in DIAGNOSTICS_LEADS],
         "mae": [_diag_cell(idx.get(("t2", mid, d)), "mae") for d in DIAGNOSTICS_LEADS],
         "n": [_diag_cell(idx.get(("t2", mid, d)), "n") for d in DIAGNOSTICS_LEADS]}
        for mid in order
    ]
    penalty = []
    for mid in order:
        row: dict = {"model_id": mid}
        for sampled, cli in DIAGNOSTICS_PENALTY_PAIRS:
            a = _diag_cell(idx.get((sampled, mid, DIAGNOSTICS_LEAD)))
            b = _diag_cell(idx.get((cli, mid, DIAGNOSTICS_LEAD)))
            row[f"{sampled}_bias"] = a
            row[f"{cli}_bias"] = b
            row[f"{sampled}_penalty"] = _clean(b - a) if a is not None and b is not None else None
            row[f"{sampled}_n"] = _diag_cell(idx.get((sampled, mid, DIAGNOSTICS_LEAD)), "n")
            row[f"{cli}_n"] = _diag_cell(idx.get((cli, mid, DIAGNOSTICS_LEAD)), "n")
        penalty.append(row)

    scope = (f"station_id={ALL_STATIONS}; window={DIAGNOSTICS_WINDOW}; "
             f"init_hour={DIAGNOSTICS_INIT:02d}Z; method={DIAGNOSTICS_METHOD}")
    return {
        **_envelope(scores, DIAGNOSTICS_WINDOW, scope=scope),
        "attribution": "None of these cuts is attributed to a cause. See METHODOLOGY §10.2 for "
                       "the three candidate explanations and the analyses that would separate "
                       "them.",
        "hourly_bias": {
            "variables": [f"t2_{h:02d}z" for h in DIAGNOSTICS_HOURS],
            "hours_utc": list(DIAGNOSTICS_HOURS),
            "lead_day": DIAGNOSTICS_LEAD,
            "rows": hourly,
        },
        "bias_by_lead": {
            "variable": "t2",
            "lead_days": list(DIAGNOSTICS_LEADS),
            "rows": by_lead,
        },
        "sampling_penalty": {
            "lead_day": DIAGNOSTICS_LEAD,
            "pairs": [{"sampled": s, "cli": c, "penalty": f"bias({c}) - bias({s})"}
                      for s, c in DIAGNOSTICS_PENALTY_PAIRS],
            "identity": "Both terms are formed from the same forecast value, so the forecast "
                        "cancels and the difference is the gap between the two truth definitions "
                        "over the days and stations that model was scored on. Read the level of "
                        "each column for a model's own diurnal behaviour and the sample sizes "
                        "for why the differences are not identical.",
            "rows": penalty,
        },
    }


def export_api(
    scores: pd.DataFrame,
    pairwise: pd.DataFrame,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
    out: str | Path | None = None,
    daily: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    instant: pd.DataFrame | None = None,
    errors: pd.DataFrame | None = None,
    status: dict | None = None,
    series_days: int = SERIES_DAYS,
    cards_out: str | Path | None = None,
) -> dict[str, int]:
    """Write the whole static JSON API.  Returns ``{relative path or glob: n files/rows}``."""
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()
    base = api_dir(out)
    base.mkdir(parents=True, exist_ok=True)
    # The per-station bundle lives next to the pages it belongs to, under /station/{ICAO}/.
    cards_base = Path(cards_out) if cards_out is not None else base.parent.parent / "station"
    written: dict[str, int] = {}

    scores = scores if scores is not None else pd.DataFrame()
    pairwise = pairwise if pairwise is not None else pd.DataFrame()

    write_json(base / "stations.json", {**_envelope(scores), "stations": _station_payload(stations)})
    write_json(base / "models.json", {**_envelope(scores), "models": _model_payload(models)})
    written["stations.json"] = len(stations) + 1
    written["models.json"] = len(models) + 1

    n_shards, index = _write_score_shards(base, scores)
    written["scores/index.json"] = 1
    written["scores/by-station/{station}.json"] = n_shards
    # Kept as an alias so an old link resolves; it is the index, not the whole table.
    write_json(base / "scores" / "latest.json", index)
    written["scores/latest.json"] = n_shards

    if len(scores):
        board = scores[scores["station_id"] == ALL_STATIONS]
    else:
        board = scores
    write_json(base / "scores" / "leaderboard.json",
               {**_envelope(scores, scope=f"station_id={ALL_STATIONS}"),
                **compact_table(_with_skill_flag(_with_permalink(board)))})
    written["scores/leaderboard.json"] = len(board)

    written["leaderboard/{view}.json"] = _write_leaderboards(base, scores)

    write_json(base / "diagnostics.json", diagnostics_payload(scores, models))
    written["diagnostics.json"] = 1

    write_json(base / "openapi.json", openapi_document())
    written["openapi.json"] = 1

    if len(pairwise):
        pw_all = pairwise[pairwise["station_id"] == ALL_STATIONS]
    else:
        pw_all = pairwise
    write_json(
        base / "pairwise" / "latest.json",
        {**_envelope(scores, scope=f"station_id={ALL_STATIONS}; per-station comparisons are in "
                                   f"scores/{{station}}/{{model}}/{{lead}}.json"),
         **compact_table(pw_all)},
    )
    written["pairwise/latest.json"] = len(pw_all)

    if errors is None and daily is None and truth is None and len(scores):
        # called without the underlying tables (e.g. from the CLI): load them for the series
        try:
            from . import store

            daily, truth = store.read_daily(), store.read_truth()
        except Exception:  # noqa: BLE001  # pragma: no cover - a missing data/ must not break the export
            daily = truth = None
    if errors is None and instant is None and len(scores):
        # the t2 series lives in the instantaneous errors, which are derived, not stored
        try:
            from . import store
            from .derive import instant_errors

            ti = store.read_truth_instant()
            if ti is not None and len(ti):
                instant = instant_errors(store.read_forecast_values(), ti, stations, models)
        except Exception:  # noqa: BLE001  # pragma: no cover - the series is not worth failing on
            instant = None
    if errors is None and daily is not None and truth is not None and len(daily) and len(truth):
        errors = error_table(daily, truth, instant)
    series = _daily_series(errors, series_days)

    n_cards, n_bundles = _write_station_cards(cards_base, scores, pairwise, series, series_days)
    written["station/{station}/cards.json"] = n_bundles
    written["cards"] = n_cards
    _prune_legacy_cards(base, stations)

    if status is not None:
        write_json(base / "status.json", status)
        written["status.json"] = 1
    return written


# ------------------------------------------------------------------------------------------
# per-station bundles
# ------------------------------------------------------------------------------------------

#: Columns that are constant inside one card's pairwise block, so they are hoisted into
#: ``pairwise_scope`` instead of repeating on every comparison.
_CARD_PAIRWISE_CONSTANTS = ("station_id", "lead_day", "window", "init_hour", "method")

#: Columns of the per-station table whose values come from a tiny fixed vocabulary.  Eight short
#: labels repeated on ten thousand rows were 39 % of the bundle; dictionary-encoding them costs
#: one line to decode and takes the file from 2.8 MB to 1.9 MB.
_DICTIONARY_COLUMNS = ("model_id", "variable", "method", "window", "model_version",
                       "segment_start", "period_start", "period_end")


def _dictionary_encode(table: dict, columns: tuple[str, ...]) -> dict:
    """Replace the values of low-cardinality columns with an index into a per-column vocabulary.

    Mutates ``table`` (whose rows were just built by :func:`compact_table` and are not shared) and
    returns ``{column: [value, …]}``.  A consumer decodes with one map per column::

        for col, values in payload["scores"]["dictionaries"].items():
            df[col] = df[col].map(dict(enumerate(values)))

    Only the per-station bundle uses this.  The leaderboards, the pairwise export and the index —
    the endpoints a casual consumer reads — stay plain ``{columns, rows}``.
    """
    dictionaries: dict[str, list] = {}
    for col in columns:
        if col not in table["columns"]:
            continue
        i = table["columns"].index(col)
        values: list = []
        seen: dict = {}
        for row in table["rows"]:
            v = row[i]
            key = (type(v).__name__, v)
            code = seen.get(key)
            if code is None:
                code = len(values)
                seen[key] = code
                values.append(v)
            row[i] = code
        dictionaries[col] = values
    return dictionaries


def _card_series(entries: list[dict]) -> tuple[list[str] | None, list[dict]]:
    """Hoist the shared date axis out of a card's per-variable series.

    All eleven variables of a card are scored on the same climatological days, so the card used to
    carry the same list of ninety dates eleven times — about half of its bytes.  The common axis
    is written once as ``series_dates``; a variable whose own axis differs still carries its own.
    """
    if not entries:
        return None, []
    common = entries[0]["dates"]
    if not all(e["dates"] == common for e in entries):
        common = None
    out = []
    for e in entries:
        item = {"variable": e["variable"], "init_hour": e["init_hour"], "method": e["method"],
                "err_c": e["err_c"]}
        if common is None or e["dates"] != common:
            item["dates"] = e["dates"]
        out.append(item)
    return common, out


def _write_station_cards(cards_dir: Path, scores: pd.DataFrame, pairwise: pd.DataFrame,
                         series: dict, series_days: int) -> tuple[int, int]:
    """One file per station: ``station/{ICAO}/cards.json``.

    Before this there was one file per station × model × lead day — about 1 800 of them, 68 MB,
    each repeating the response envelope, the units block and the method block, and each carrying
    a slice of a table that was *also* published whole in ``scores/by-station/{station}.json``.
    Merging them removes that duplication outright: the station's scores are written once, in the
    same compact encoding as before, and each card adds only what is its own — its pairwise
    comparisons and its daily error series.  A permanent-link page links to its card by fragment,
    ``/station/KNYC/cards.json#gfs-1``, and ``cards_by_id`` maps that fragment to the card.
    """
    n_cards = n_files = 0
    if scores is None or len(scores) == 0:
        return 0, 0
    pw_idx: dict[tuple[str, int], pd.DataFrame] = {}
    if pairwise is not None and len(pairwise):
        for key, grp in pairwise.groupby(["station_id", "lead_day"], observed=True):
            pw_idx[(key[0], int(key[1]))] = grp

    for station_id, st_grp in scores.groupby("station_id", observed=True):
        station_id = str(station_id)
        table = compact_table(_with_skill_flag(st_grp).drop(columns=["station_id"]))
        table["dictionaries"] = _dictionary_encode(table, _DICTIONARY_COLUMNS)
        cards = []
        for (mid, lead), grp in st_grp.groupby(["model_id", "lead_day"], observed=True):
            lead = int(lead)
            pw_grp = pw_idx.get((station_id, lead))
            if pw_grp is not None and len(pw_grp):
                mask = (pw_grp["model_a"] == mid) | (pw_grp["model_b"] == mid)
                # Headline slice only: the same comparison otherwise appears 32 times over
                # (window × init × method) and again in the other model's card.  The complete
                # table is one download away at pairwise/latest.json.
                mask &= (pw_grp["window"] == CARD_PAIRWISE_WINDOW)
                mask &= pw_grp["init_hour"].astype(int) == SERIES_INIT
                mask &= pw_grp["method"] == SERIES_METHOD
                pw_grp = pw_grp[mask]
                drop = _ENVELOPE_CONSTANTS + tuple(
                    c for c in _CARD_PAIRWISE_CONSTANTS if c in pw_grp.columns)
                pw_table = compact_table(pw_grp, drop=drop)
            else:
                pw_table = {"columns": [], "rows": []}
            versions = sorted({str(v) for v in grp.get("model_version", pd.Series(dtype=str))
                               .dropna().unique()})
            segments = sorted({str(v) for v in grp.get("segment_start", pd.Series(dtype=str))
                               .dropna().unique()})
            dates, entries = _card_series(series.get((station_id, mid, lead), []))
            cards.append({
                "id": f"{mid}-{lead}",
                "model_id": mid,
                "lead_day": lead,
                "model_version": versions[0] if len(versions) == 1 else versions,
                "segment_start": segments[0] if len(segments) == 1 else segments,
                "permalink": permalink(station_id, mid, lead),
                "pairwise": pw_table,
                "series_dates": dates,
                "series": entries,
            })
            n_cards += 1
        payload = {
            **_envelope(scores, scope=f"station_id={station_id}"),
            "kind": "station-cards",
            "station_id": station_id,
            "note": "every published aggregate for this station, plus one card per model and lead "
                    "day. A permanent-link page addresses its card by fragment, e.g. "
                    f"/station/{station_id}/cards.json#gfs-1; scores rows are matched on "
                    "(model_id, lead_day). The columns listed in scores.dictionaries hold an "
                    "integer index into that column's vocabulary, not the value itself.",
            "permalink_template": "/station/{station_id}/model/{model_id}/lead/{lead_day}/",
            "scores": table,
            "n_scores": len(table["rows"]),
            "pairwise_scope": {"window": CARD_PAIRWISE_WINDOW, "init_hour": SERIES_INIT,
                               "method": SERIES_METHOD, "station_id": station_id,
                               "note": "every window/init/method is in pairwise/latest.json"},
            "series_days": series_days,
            "series_scope": {"init_hour": SERIES_INIT, "method": SERIES_METHOD,
                             "note": "other init/method slices and the matching forecast and "
                                     "observation values are in "
                                     "/data/daily_errors/{station}.csv.gz"},
            "cards": cards,
        }
        write_json(cards_dir / station_id / "cards.json", payload)
        n_files += 1
    return n_cards, n_files


def _prune_legacy_cards(base: Path, stations: list[Station]) -> None:
    """Delete the pre-merge ``scores/{station}/{model}/{lead}.json`` tree from an older build."""
    scores_dir = base / "scores"
    if not scores_dir.is_dir():
        return
    keep = {"by-station"}
    for child in scores_dir.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)


# ------------------------------------------------------------------------------------------
# sharded scores export
# ------------------------------------------------------------------------------------------

def _write_score_shards(base: Path, scores: pd.DataFrame) -> tuple[int, dict]:
    """The index of the per-station export, plus a pointer at the historical shard path.

    The scores table crossed 20 MB in August 2026 and Cloudflare Pages refuses a single file above
    25 MiB, so it has always been published one file per station.  Since v0.3.2 that file is
    ``/station/{ICAO}/cards.json`` — the same rows, in the same compact encoding, in the same file
    as the per-card blocks that used to repeat them.  ``scores/by-station/{station}.json`` stays
    as a small pointer so an existing link resolves and says where the data went, instead of
    404-ing or, worse, serving a second copy of 65 MB.

    ``permalink`` is not a column of the per-station table: the envelope carries
    ``permalink_template`` and every row has ``model_id`` and ``lead_day``, which is the same
    information without 40 bytes on each of a quarter of a million rows.  The pre-ranked
    leaderboards, which are what a casual consumer reads, still carry a literal permalink.
    """
    shard_dir = base / "scores" / "by-station"
    for stale in shard_dir.glob("*.json"):
        stale.unlink()
    shards = []
    columns: list[str] = []
    if scores is not None and len(scores):
        columns = [c for c in _with_skill_flag(scores).columns
                   if c not in _ENVELOPE_CONSTANTS and c != "station_id"]
        for station_id, grp in scores.groupby("station_id", observed=True):
            href = f"/station/{station_id}/cards.json"
            shards.append({"station_id": str(station_id),
                           "path": f"station/{station_id}/cards.json",
                           "href": href,
                           "rows": int(len(grp))})
            write_json(shard_dir / f"{station_id}.json", {
                **_envelope(scores, scope=f"station_id={station_id}"),
                "station_id": str(station_id),
                "moved_to": href,
                "note": "the per-station scores now live in the station's card bundle, together "
                        "with the pairwise comparisons and daily error series that used to "
                        "duplicate them; read `scores` in that document.",
                "columns": columns,
                "rows": [],
                "n_rows": int(len(grp)),
            })
    index = {
        **_envelope(scores),
        "index_of": "scores",
        "note": "the scores table is published one file per station, as the `scores` block of "
                "/station/{ICAO}/cards.json; station_id is dropped from each row and carried in "
                "the envelope, and the permanent link of a row is permalink_template filled in "
                "with its model_id and lead_day",
        "permalink_template": "/station/{station_id}/model/{model_id}/lead/{lead_day}/",
        "columns": columns,
        "rows": [],
        "n_rows": int(len(scores)) if scores is not None else 0,
        "shards": shards,
        "available": _available(scores),
        "leaderboards": [f"leaderboard/{w}-{int(i):02d}z-{m}-{v}.json"
                         for w, i, m, v in LEADERBOARD_VIEWS],
    }
    write_json(base / "scores" / "index.json", index)
    return len(shards), index


def _available(scores: pd.DataFrame) -> dict:
    """Which combinations actually exist, so a consumer does not have to fetch to find out."""
    if scores is None or len(scores) == 0:
        return {}
    def uniq(col, cast=str):
        if col not in scores.columns:
            return []
        return sorted({cast(v) for v in scores[col].dropna().unique()})
    return {
        "stations": uniq("station_id"),
        "models": uniq("model_id"),
        "variables": uniq("variable"),
        "methods": uniq("method"),
        "windows": uniq("window"),
        "init_hours": uniq("init_hour", int),
        "lead_days": uniq("lead_day", int),
    }


# ------------------------------------------------------------------------------------------
# pre-built leaderboards and the OpenAPI description
# ------------------------------------------------------------------------------------------

_BOARD_FIELDS = (
    "model_id", "n", "mae", "mae_ci_low", "mae_ci_high", "bias", "bias_ci_low", "bias_ci_high",
    "rmse", "hit1f", "hit2f", "hit3f", "skill_persistence", "period_start", "period_end",
)


def _write_leaderboards(base: Path, scores: pd.DataFrame) -> int:
    """One ranked file per site view: ``leaderboard/{window}-{init}z-{method}-{variable}.json``.

    Results are objects (not the compact encoding) because these files are small, are what a
    third party is most likely to read, and every row needs its ``permalink`` and its rank next to
    the numbers.  Groups below :data:`MIN_N` are included but carry ``"rank": null`` and
    ``"ranked": false``, exactly as the site greys them out.
    """
    n = 0
    for window, init_hour, method, variable in LEADERBOARD_VIEWS:
        rows: list[dict] = []
        if scores is not None and len(scores):
            sub = scores[
                (scores["station_id"] == ALL_STATIONS)
                & (scores["window"] == window)
                & (scores["init_hour"].astype(int) == int(init_hour))
                & (scores["method"] == method)
                & (scores["variable"] == variable)
            ].sort_values(["lead_day", "mae"])
            rank_by_lead: dict[int, int] = {}
            for _, r in sub.iterrows():
                lead = int(r["lead_day"])
                ranked = int(r["n"]) >= MIN_N and r["model_id"] != PERSISTENCE_ID
                rank = None
                if ranked:
                    rank = rank_by_lead.get(lead, 0) + 1
                    rank_by_lead[lead] = rank
                item = {"lead_day": lead, "rank": rank, "ranked": ranked,
                        "baseline": r["model_id"] == PERSISTENCE_ID}
                item.update({k: _clean(r[k]) for k in _BOARD_FIELDS if k in sub.columns})
                n_common = _clean(r.get("n_common"))
                item["n_common"] = n_common
                item["skill_reliable"] = bool(
                    n_common is not None and n_common >= SKILL_MIN_COMMON
                    and _clean(r.get("skill_persistence")) is not None)
                item["permalink"] = permalink(ALL_STATIONS, r["model_id"], lead)
                rows.append(item)
        payload = {
            **_envelope(scores, window=window,
                        scope=f"station_id={ALL_STATIONS}"),
            "view": {"window": window, "init_hour": int(init_hour), "method": method,
                     "variable": variable,
                     "page": f"{SITE}/v/{window}-{int(init_hour):02d}z-{method}-{variable}/"},
            "results": rows,
        }
        write_json(base / "leaderboard"
                   / f"{window}-{int(init_hour):02d}z-{method}-{variable}.json", payload)
        n += 1
    return n


def openapi_document() -> dict:
    """A minimal but valid OpenAPI 3.1 description of the static API."""
    envelope = {
        "type": "object",
        "required": ["schema_version", "generated_at", "data_through", "window", "units",
                     "method", "truth"],
        "properties": {
            "schema_version": {"type": "string"},
            "methodology_version": {"type": "string"},
            "castcheck_version": {"type": "string"},
            "generated_at": {"type": "string", "format": "date-time"},
            "data_through": {"type": ["string", "null"], "format": "date"},
            "next_update": {"type": "string", "format": "date-time"},
            "window": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "examples": ["30d", "90d", "365d", "all"]},
                    "days": {"type": ["integer", "null"]},
                    "start": {"type": ["string", "null"], "format": "date"},
                    "end": {"type": ["string", "null"], "format": "date"},
                },
            },
            "units": {"type": "object", "additionalProperties": {"type": "string"}},
            "method": {"type": "object"},
            "truth": {"type": "object"},
            "license": {"type": "string"},
        },
    }
    table = {
        "allOf": [
            {"$ref": "#/components/schemas/Envelope"},
            {"type": "object",
             "properties": {
                 "columns": {"type": "array", "items": {"type": "string"}},
                 "rows": {"type": "array", "items": {"type": "array"}},
             }},
        ]
    }

    def get(summary: str, schema_ref: str, params: list | None = None,
            servers: list | None = None) -> dict:
        op = {
            "summary": summary,
            "responses": {"200": {"description": "OK", "content": {
                "application/json": {"schema": {"$ref": schema_ref}}}}},
        }
        if params:
            op["parameters"] = params
        if servers:
            # The per-station bundle is served from /station/{ICAO}/, not from /api/v1: without
            # its own server the document's base URL composes a path that does not exist.
            op["servers"] = servers
        return {"get": op}

    def path_param(name: str, example, description: str) -> dict:
        return {"name": name, "in": "path", "required": True,
                "schema": {"type": "string"}, "example": example,
                "description": description}

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "CastCheck API",
            "version": f"1.{SCHEMA_VERSION}",
            "summary": "Station-level verification of raw weather-model 2 m temperature forecasts.",
            "description": (
                "Static JSON on a CDN: no keys, no rate limit, CORS open, cached one hour. "
                "MAE, bias, RMSE and interval bounds are in degrees Celsius; the HTML site "
                "displays the same differences in degrees Fahrenheit. Groups with fewer than "
                f"{MIN_N} scored days are published but never ranked."
            ),
            "license": {"name": "CC BY 4.0",
                        "url": "https://creativecommons.org/licenses/by/4.0/"},
            "contact": {"url": SITE},
        },
        "servers": [{"url": f"{SITE}/api/v1"}],
        "paths": {
            "/scores/index.json": get(
                "Index of the sharded scores export: the shard of every station, the column "
                "list, and which stations, models, variables, windows and leads exist.",
                "#/components/schemas/Table"),
            "/scores/latest.json": get(
                "Alias of /scores/index.json. Before methodology v0.3 this served the whole "
                "scores table in one file; it is now the index of the per-station shards.",
                "#/components/schemas/Table"),
            "/scores/by-station/{station}.json": get(
                "Pointer at the station's bundle: `moved_to` names the file that carries the "
                "rows. Kept so that an existing link resolves.",
                "#/components/schemas/Envelope",
                [path_param("station", "KNYC",
                            "ICAO identifier, or ALL for the aggregate")]),
            "/scores/leaderboard.json": get(
                "The station_id=ALL slice of the scores table.",
                "#/components/schemas/Table"),
            "/leaderboard/{view}.json": get(
                "One pre-ranked leaderboard per site view, e.g. 90d-00z-bilinear-t2.",
                "#/components/schemas/Leaderboard",
                [path_param("view", "90d-00z-bilinear-t2",
                            "{window}-{init}z-{method}-{variable}; window is one of "
                            f"{'/'.join(WINDOWS)}, variable one of {'/'.join(VARIABLES)}")]),
            "/station/{station}/cards.json": get(
                "The per-station bundle: every published aggregate for the station under "
                "`scores`, and one entry in `cards` per model and lead day with that card's "
                "pairwise comparisons and daily error series. A permanent-link page addresses "
                "its card by fragment, e.g. /station/KNYC/cards.json#gfs-1. This one file is "
                "served from the site root, not from /api/v1.",
                "#/components/schemas/Card",
                [path_param("station", "KNYC",
                            "ICAO identifier, or ALL for the aggregate")],
                servers=[{"url": SITE}]),
            "/pairwise/latest.json": get(
                "Paired model-vs-model MAE differences on common days.",
                "#/components/schemas/Table"),
            "/stations.json": get("Station metadata.", "#/components/schemas/Envelope"),
            "/models.json": get("Model registry.", "#/components/schemas/Envelope"),
            "/status.json": get(
                "Pipeline completeness report. Unlike every score response this one is not an "
                "envelope: it has no window, units, method or truth block, because it describes "
                "which runs and observations arrived, not a score.",
                "#/components/schemas/Status"),
            "/diagnostics.json": get(
                "Diurnal-structure diagnostic for station_id=ALL over the 90d, 00Z, bilinear "
                "slice: per-model bias at each of the four synoptic instants, bias against lead "
                "day, and the sampling penalty bias(*_cli) - bias(*_s). Reported without "
                "attribution (METHODOLOGY §10.2).",
                "#/components/schemas/Diagnostics"),
        },
        "components": {"schemas": {
            "Envelope": envelope,
            "Status": {
                "type": "object",
                "description": "Data-completeness report (see castcheck/status.py). Additional "
                               "properties are permitted: the report grows as the pipeline does.",
                "properties": {
                    "schema_version": {"type": "string"},
                    "methodology_version": {"type": "string"},
                    "generated_at": {"type": "string", "format": "date-time"},
                    "as_of": {"type": ["string", "null"], "format": "date"},
                    "days": {"type": "integer"},
                    "ok": {"type": "boolean"},
                    "n_stations": {"type": "integer"},
                    "n_gaps": {"type": "integer"},
                    "n_current_gaps": {"type": "integer"},
                    "n_pending": {"type": "integer"},
                    "gaps_today": {"type": "array", "items": {"type": "string"}},
                    "models": {"type": "array", "items": {"type": "object"}},
                    "truth": {"type": "array", "items": {"type": "object"}},
                    "uptime": {"type": "object"},
                },
            },
            "Table": table,
            "Leaderboard": {
                "allOf": [
                    {"$ref": "#/components/schemas/Envelope"},
                    {"type": "object", "properties": {
                        "view": {"type": "object"},
                        "results": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "lead_day": {"type": "integer"},
                                "rank": {"type": ["integer", "null"]},
                                "ranked": {"type": "boolean"},
                                "model_id": {"type": "string"},
                                "n": {"type": "integer"},
                                "mae": {"type": ["number", "null"]},
                                "mae_ci_low": {"type": ["number", "null"]},
                                "mae_ci_high": {"type": ["number", "null"]},
                                "bias": {"type": ["number", "null"]},
                                "permalink": {"type": "string"},
                            },
                        }},
                    }},
                ]
            },
            "Diagnostics": {
                "allOf": [
                    {"$ref": "#/components/schemas/Envelope"},
                    {"type": "object", "properties": {
                        "scope": {"type": "string"},
                        "attribution": {"type": "string"},
                        "hourly_bias": {"type": "object", "properties": {
                            "variables": {"type": "array", "items": {"type": "string"}},
                            "hours_utc": {"type": "array", "items": {"type": "integer"}},
                            "lead_day": {"type": "integer"},
                            "rows": {"type": "array", "items": {"type": "object"}},
                        }},
                        "bias_by_lead": {"type": "object", "properties": {
                            "variable": {"type": "string"},
                            "lead_days": {"type": "array", "items": {"type": "integer"}},
                            "rows": {"type": "array", "items": {"type": "object"}},
                        }},
                        "sampling_penalty": {"type": "object", "properties": {
                            "lead_day": {"type": "integer"},
                            "pairs": {"type": "array", "items": {"type": "object"}},
                            "rows": {"type": "array", "items": {"type": "object"}},
                        }},
                    }},
                ]
            },
            "Card": {
                "allOf": [
                    {"$ref": "#/components/schemas/Envelope"},
                    {"type": "object", "properties": {
                        "station_id": {"type": "string"},
                        "permalink_template": {"type": "string"},
                        "scores": {"$ref": "#/components/schemas/Table"},
                        "cards": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "model_id": {"type": "string"},
                                "lead_day": {"type": "integer"},
                                "permalink": {"type": "string"},
                                "pairwise": {"$ref": "#/components/schemas/Table"},
                                "series_dates": {"type": ["array", "null"],
                                                 "items": {"type": "string"}},
                                "series": {"type": "array", "items": {"type": "object"}},
                            },
                        }},
                    }},
                ]
            },
        }},
    }
