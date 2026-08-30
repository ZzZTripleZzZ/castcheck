"""Data-completeness report (DESIGN §5, §6): what is missing, right now.

``build()`` answers three questions for the last ``days`` (default 14) days:

1. for every ``model × initialization``, did a **complete** run arrive?  A run is complete when every
   station has a present ``t2`` value at every expected forecast step (``max_h / step_h`` steps,
   f000 excluded — METHODOLOGY §2.2 and the AIWP fill-value note in DESIGN §4);
2. for every station, is there a first-final NWS CLI value for each climatological day;
3. what is the newest initialization we hold per model.

``exit_code()`` implements the CLI contract: **non-zero when something that should already exist for
the current day is missing**, so that the scheduled workflow fails loudly instead of silently
publishing a hole.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION, __version__
from .config import PUBLIC_DIR, ModelSpec, Station, load_models, load_stations

__all__ = ["DEFAULT_DAYS", "EXIT_GAPS", "EXIT_OK", "build", "exit_code", "write_status"]

EXIT_OK = 0
EXIT_GAPS = 1
DEFAULT_DAYS = 14


def _expected_steps(model: ModelSpec) -> int:
    """Forecast steps a complete run must carry (f000 excluded)."""
    step = max(int(model.step_h), 1)
    return max(int(model.max_h) // step, 0)


def _as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def build(
    as_of: date | str | None = None,
    days: int = DEFAULT_DAYS,
    values: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
) -> dict:
    """Build the completeness report.  Reads ``data/`` unless ``values``/``truth`` are supplied."""
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()
    as_of_d = _as_date(as_of) if as_of is not None else datetime.now(UTC).date()
    window = [as_of_d - timedelta(days=i) for i in range(days - 1, -1, -1)]
    station_ids = [s.id for s in stations]

    if values is None or truth is None:
        from . import store
        if values is None:
            values = store.read_forecast_values(start=(window[0] - timedelta(days=1)).isoformat())
        if truth is None:
            years = sorted({d.year for d in window})
            truth = store.read_truth(years)

    report: dict = {
        "castcheck_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "as_of": as_of_d.isoformat(),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "days": days,
        "dates": [d.isoformat() for d in window],
        "n_stations": len(stations),
        "models": [],
        "truth": [],
        "gaps": [],
        "current_gaps": [],
    }

    # ---- model runs -----------------------------------------------------------------------
    present = pd.DataFrame()
    if values is not None and len(values):
        v = values.copy()
        v["init_time"] = pd.to_datetime(v["init_time"], utc=True)
        if "missing_reason" not in v:
            v["missing_reason"] = ""
        present = v[
            (v["variable"] == "t2")
            & (v["missing_reason"].fillna("") == "")
            & v["value_c"].notna()
            & (v.get("method", "bilinear") == "bilinear")
        ]

    counts = {}
    latest_init: dict[str, str | None] = {}
    if len(present):
        g = (
            present.groupby(["model_id", "init_time", "station_id"], observed=True)["valid_time"]
            .nunique()
        )
        counts = {(m, pd.Timestamp(i), s): int(n) for (m, i, s), n in g.items()}
        for model_id, grp in present.groupby("model_id", observed=True):
            latest_init[model_id] = pd.Timestamp(grp["init_time"].max()).isoformat()

    for m in models:
        exp = _expected_steps(m)
        for init_hour in m.inits:
            day_rows = []
            for d in window:
                init = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=int(init_hour))
                per_station = [counts.get((m.model_id, init, s), 0) for s in station_ids]
                n_complete = sum(1 for c in per_station if c >= exp)
                n_any = sum(1 for c in per_station if c > 0)
                complete = n_complete == len(station_ids) and len(station_ids) > 0
                day_rows.append({
                    "date": d.isoformat(),
                    "init_time": init.isoformat(),
                    "complete": bool(complete),
                    "stations_complete": n_complete,
                    "stations_any": n_any,
                    "expected_steps": exp,
                    "values": int(sum(per_station)),
                })
                if not complete:
                    gap = {
                        "type": "model_run",
                        "model_id": m.model_id,
                        "init_hour": int(init_hour),
                        "date": d.isoformat(),
                        "detail": f"{n_complete}/{len(station_ids)} stations with {exp} steps",
                    }
                    report["gaps"].append(gap)
                    if d == as_of_d:
                        report["current_gaps"].append(gap)
            report["models"].append({
                "model_id": m.model_id,
                "family": m.family,
                "init_hour": int(init_hour),
                "expected_steps": exp,
                "latest_init": latest_init.get(m.model_id),
                "days": day_rows,
                "n_complete": sum(1 for r in day_rows if r["complete"]),
                "n_missing": sum(1 for r in day_rows if not r["complete"]),
            })

    # ---- truth ----------------------------------------------------------------------------
    final: set[tuple[str, date]] = set()
    partial: set[tuple[str, date]] = set()
    if truth is not None and len(truth):
        t = truth.copy()
        t["climo_date"] = pd.to_datetime(t["climo_date"]).dt.date
        if "is_final" not in t:
            t["is_final"] = False
        cli = t[(t["source"] == "CLI") & t["is_final"].fillna(False).astype(bool)]
        for st, d, tmax, tmin in zip(cli["station_id"], cli["climo_date"], cli["tmax_c"], cli["tmin_c"]):
            ok = pd.notna(tmax) and pd.notna(tmin)
            (final if ok else partial).add((st, d))
        for st, d in zip(t["station_id"], t["climo_date"]):
            if (st, d) not in final:
                partial.add((st, d))

    truth_deadline = as_of_d - timedelta(days=1)  # yesterday's CLI is the newest that can exist
    for s in stations:
        day_rows = []
        for d in window:
            has = (s.id, d) in final
            day_rows.append({"date": d.isoformat(), "cli_final": bool(has),
                             "any_source": bool(has or (s.id, d) in partial)})
            if not has and d <= truth_deadline:
                gap = {"type": "truth", "station_id": s.id, "date": d.isoformat(),
                       "detail": "no first-final CLI with both tmax and tmin"}
                report["gaps"].append(gap)
                if d == truth_deadline:
                    report["current_gaps"].append(gap)
        report["truth"].append({
            "station_id": s.id, "name": s.name, "cli_pil": s.cli_pil, "days": day_rows,
            "n_missing": sum(1 for r in day_rows if not r["cli_final"] and r["date"] <= truth_deadline.isoformat()),
        })

    report["n_gaps"] = len(report["gaps"])
    report["n_current_gaps"] = len(report["current_gaps"])
    report["gaps_today"] = report["current_gaps"]  # alias used by the CLI
    report["ok"] = report["n_current_gaps"] == 0
    return report


def exit_code(report: dict) -> int:
    """``0`` when nothing that should already exist today is missing, ``1`` otherwise."""
    return EXIT_OK if report.get("n_current_gaps", 0) == 0 else EXIT_GAPS


def status_path(out: str | Path | None = None) -> Path:
    return Path(out) if out is not None else PUBLIC_DIR / "api" / "v1" / "status.json"


def write_status(
    as_of: date | str | None = None,
    out: str | Path | None = None,
    report: dict | None = None,
    **kwargs,
) -> dict:
    """Build (unless given) and write ``public/api/v1/status.json``.  Returns the report.

    ``report["ok"]`` and :func:`exit_code` carry the CLI contract; ``report["path"]`` is where the
    JSON was written.
    """
    from .api import write_json

    report = report if report is not None else build(as_of=as_of, **kwargs)
    path = status_path(out)
    write_json(path, report)
    report = dict(report)
    report["path"] = str(path)
    return report
