"""Static JSON API export (DESIGN §6).

Everything under ``public/api/v1`` is a plain file; there is no server.  Large tables use a compact
``{"columns": [...], "rows": [[...]]}`` encoding, constant columns (versions, ``computed_at``) are
lifted into the envelope, and floats are rounded so the payload stays small enough to be served
from a CDN and read by the ~4 KB chart script.

Endpoints
---------
``stations.json``                              station metadata
``models.json``                                model metadata (incl. the persistence baseline)
``scores/latest.json``                         the full ``scores`` table
``scores/leaderboard.json``                    the ``station_id="ALL"`` slice only (small; used by ``/``)
``scores/{station}/{model}/{lead}.json``        one permanent-link card: every window/init/method/
                                               variable for that combination, its pairwise
                                               comparisons, and the last 90 days of daily errors
``pairwise/latest.json``                       the ``station_id="ALL"`` pairwise slice
``status.json``                                data-completeness report (see ``status.py``)
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION, __version__
from .config import PUBLIC_DIR, ModelSpec, Station, load_models, load_stations
from .verify import ALL_STATIONS, PERSISTENCE_ID, error_table

__all__ = ["api_dir", "compact_table", "export_api", "write_json"]

SERIES_DAYS = 90
_ENVELOPE_CONSTANTS = ("computed_at", "methodology_version", "schema_version")


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
        return None if math.isnan(v) else round(v, 4)
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


def _envelope(scores: pd.DataFrame, **extra) -> dict:
    computed_at = ""
    if scores is not None and len(scores) and "computed_at" in scores:
        computed_at = str(scores["computed_at"].iloc[0])
    env = {
        "castcheck_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "computed_at": computed_at or pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "license": "CC-BY-4.0 (CastCheck derived data); see /data/ for upstream licences",
    }
    env.update(extra)
    return env


def _station_payload(stations: list[Station]) -> list[dict]:
    out = [{
        "id": s.id, "name": s.name, "cli_pil": s.cli_pil, "tz": s.tz,
        "std_offset_h": s.std_offset_h, "lat": s.lat, "lon": s.lon, "elev_m": s.elev_m,
    } for s in stations]
    out.append({"id": ALL_STATIONS, "name": "All stations (mean of daily station errors)",
                "cli_pil": None, "tz": None, "std_offset_h": None,
                "lat": None, "lon": None, "elev_m": None})
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
    """(station_id, model_id, lead_day) → list of per-(init,method,variable) error series."""
    if errors is None or len(errors) == 0:
        return {}
    e = errors.copy()
    e["climo_date"] = pd.to_datetime(e["climo_date"])
    cutoff = e["climo_date"].max() - pd.Timedelta(days=series_days - 1)
    e = e[e["climo_date"] >= cutoff]
    out: dict[tuple, list[dict]] = {}
    keys = ["station_id", "model_id", "lead_day", "init_hour", "method", "variable"]
    for key, grp in e.sort_values("climo_date").groupby(keys, observed=True):
        st, mid, lead, init_hour, method, variable = key
        out.setdefault((st, mid, int(lead)), []).append({
            "init_hour": int(init_hour), "method": method, "variable": variable,
            "dates": [d.date().isoformat() for d in grp["climo_date"]],
            "err_c": [_clean(v) for v in grp["err"]],
            "obs_c": [_clean(v) for v in grp["obs_c"]],
            "fcst_c": [_clean(v) for v in grp["fcst_c"]],
        })
    return out


def export_api(
    scores: pd.DataFrame,
    pairwise: pd.DataFrame,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
    out: str | Path | None = None,
    daily: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    errors: pd.DataFrame | None = None,
    status: dict | None = None,
    series_days: int = SERIES_DAYS,
) -> dict[str, int]:
    """Write the whole static JSON API.  Returns ``{relative path or glob: n files/rows}``."""
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()
    base = api_dir(out)
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    scores = scores if scores is not None else pd.DataFrame()
    pairwise = pairwise if pairwise is not None else pd.DataFrame()

    write_json(base / "stations.json", {**_envelope(scores), "stations": _station_payload(stations)})
    write_json(base / "models.json", {**_envelope(scores), "models": _model_payload(models)})
    written["stations.json"] = len(stations) + 1
    written["models.json"] = len(models) + 1

    write_json(base / "scores" / "latest.json", {**_envelope(scores), **compact_table(scores)})
    written["scores/latest.json"] = len(scores)

    if len(scores):
        board = scores[scores["station_id"] == ALL_STATIONS]
    else:
        board = scores
    write_json(base / "scores" / "leaderboard.json",
               {**_envelope(scores, scope=f"station_id={ALL_STATIONS}"), **compact_table(board)})
    written["scores/leaderboard.json"] = len(board)

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
    if errors is None and daily is not None and truth is not None and len(daily) and len(truth):
        errors = error_table(daily, truth)
    series = _daily_series(errors, series_days)

    n_cards = 0
    if len(scores):
        pw_idx = {}
        if len(pairwise):
            for key, grp in pairwise.groupby(["station_id", "lead_day"], observed=True):
                pw_idx[(key[0], int(key[1]))] = grp
        for (st, mid, lead), grp in scores.groupby(["station_id", "model_id", "lead_day"], observed=True):
            lead = int(lead)
            pw_grp = pw_idx.get((st, lead))
            if pw_grp is not None and len(pw_grp):
                mask = (pw_grp["model_a"] == mid) | (pw_grp["model_b"] == mid)
                pw_grp = pw_grp[mask]
            payload = {
                **_envelope(scores),
                "station_id": st, "model_id": mid, "lead_day": lead,
                "permalink": f"/station/{st}/model/{mid}/lead/{lead}/",
                "scores": compact_table(grp),
                "pairwise": compact_table(pw_grp) if pw_grp is not None else {"columns": [], "rows": []},
                "series_days": series_days,
                "series": series.get((st, mid, lead), []),
            }
            write_json(base / "scores" / str(st) / str(mid) / f"{lead}.json", payload)
            n_cards += 1
    written["scores/{station}/{model}/{lead}.json"] = n_cards

    if status is not None:
        write_json(base / "status.json", status)
        written["status.json"] = 1
    return written
