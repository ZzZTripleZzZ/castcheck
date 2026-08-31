"""Data-completeness report (DESIGN §5, §6): what is missing, right now.

``build()`` answers three questions for the last ``days`` (default 90 — the GitHub-Status-style
uptime window the site draws) days:

1. for every ``model × initialization``, did a **complete** run arrive?  A run is complete when every
   station has a present ``t2`` value at every expected forecast step (``max_h / step_h`` steps,
   f000 excluded — METHODOLOGY §2.2 and the AIWP fill-value note in DESIGN §4);
2. for every station, is there a first-final NWS CLI value for each climatological day;
3. for every station, are all four synoptic instants present in ``truth_instant`` — the truth the
   headline ``t2`` score is computed against, so a hole there silently shrinks the headline;
4. what is the newest initialization we hold per model.

Three kinds of day are **not** downtime and are excluded from the uptime denominator:

* days before a model's own ``period_start`` — the first initialization CastCheck ever held for it.
  A model that entered the record three weeks ago cannot have been "down" for the 69 days before
  (review B8);
* AIWP initializations that the upstream archive never produced.  The NOAA/CIRA 0.25° bucket
  publishes the GFS-initialized models on alternating cycles (00Z one day, 12Z the next), so the
  nominal 2 runs/day is not the upstream contract.  ``AiwpSource.available_inits`` is asked what
  actually exists and the rest are marked ``not produced upstream``;
* **runs and reports that are not due yet.**  A 12Z ECMWF run does not exist at 13 UTC, and the
  first final CLI for a Los Angeles climatological day does not exist at 06 UTC — local midnight
  there has not happened.  The deadlines come from :mod:`castcheck.schedule`, the same module the
  fetcher plans from, so the status page can never call a run missing that ``fetch-latest`` has
  quite correctly not asked for.  These days are drawn grey (``not_due_yet``) and are excluded from
  both the gap list and the uptime denominator.

``report["ok"]`` — and the red bar the site draws from it — therefore mean "everything **due** for
the current day is present", which is what an operator wants to know at 07 UTC.

``exit_code()`` implements the CLI contract: **non-zero when something that should already exist for
the current day is missing**, so that the scheduled workflow fails loudly instead of silently
publishing a hole.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import METHODOLOGY_VERSION, SCHEMA_VERSION, __version__, schedule
from .config import PUBLIC_DIR, ModelSpec, Station, load_models, load_stations

__all__ = ["DEFAULT_DAYS", "EXIT_GAPS", "EXIT_OK", "MAX_LISTED_GAPS", "build", "exit_code",
           "write_status"]

EXIT_OK = 0
EXIT_GAPS = 1
DEFAULT_DAYS = 90  # the uptime window drawn on /status/
#: The four common synoptic instants (METHODOLOGY §2.2); a truth_instant day is complete at four.
INSTANT_HOURS = (0, 6, 12, 18)
MAX_LISTED_GAPS = 400  # `gaps` is a convenience list; the per-day grids are the full record


def _expected_steps(model: ModelSpec) -> int:
    """Forecast steps a complete run must carry (f000 excluded).

    Counted as the number of steps with ``lead_h >= step_h``: ``max_h / step_h``.  The count of
    *rows* is not usable here because AIWP files carry the analysis step f000 as well, so a run
    missing f240 would otherwise still reach 41 values and be scored complete.
    """
    step = max(int(model.step_h), 1)
    return max(int(model.max_h) // step, 0)


def _upstream_inits(models: list[ModelSpec], start: date, end: date) -> dict[str, set]:
    """``model_id -> {init_time}`` actually present upstream, for the sources that publish gaps.

    Only AIWP is asked (its 0.25° archive produces the GFS-initialized models on alternating
    cycles).  Any failure — no network, no bucket, an import error — returns nothing for that
    model, which means "no upstream information" and leaves the old behaviour in place.
    """
    out: dict[str, set] = {}
    aiwp = [m for m in models if m.source == "aiwp"]
    if not aiwp:
        return out
    try:
        from .sources.aiwp import AiwpSource

        src = AiwpSource()
    except Exception:  # noqa: BLE001 - status must never fail because a source is unavailable
        return out
    for m in aiwp:
        try:
            inits = src.available_inits(m, start, end)
        except Exception:  # noqa: BLE001
            continue
        if inits:
            out[m.model_id] = {pd.Timestamp(t).tz_convert("UTC")
                               if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")
                               for t in inits}
    return out


def _as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def build(
    as_of: date | str | None = None,
    days: int = DEFAULT_DAYS,
    values: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    truth_instant: pd.DataFrame | None = None,
    stations: list[Station] | None = None,
    models: list[ModelSpec] | None = None,
    upstream: bool = True,
    now: datetime | None = None,
) -> dict:
    """Build the completeness report.  Reads ``data/`` unless ``values``/``truth`` are supplied.

    ``now`` is the instant every "is this due yet?" question is asked against; it defaults to the
    real clock and is passed explicitly by the tests.
    """
    stations = list(stations) if stations is not None else load_stations()
    models = list(models) if models is not None else load_models()
    now = now or schedule.now_utc()
    if now.tzinfo is None:  # pragma: no cover - defensive
        now = now.replace(tzinfo=UTC)
    as_of_d = _as_date(as_of) if as_of is not None else now.date()
    window = [as_of_d - timedelta(days=i) for i in range(days - 1, -1, -1)]
    station_ids = [s.id for s in stations]

    if values is None or truth is None or truth_instant is None:
        from . import store
        if values is None:
            values = store.read_forecast_values(start=(window[0] - timedelta(days=1)).isoformat())
        years = sorted({d.year for d in window})
        if truth is None:
            truth = store.read_truth(years)
        if truth_instant is None:
            try:
                truth_instant = store.read_truth_instant(years)
            except Exception:  # noqa: BLE001 - the table may not exist in an older checkout
                truth_instant = None

    report: dict = {
        "castcheck_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "as_of": as_of_d.isoformat(),
        "now": now.replace(microsecond=0).isoformat(),
        "generated_at": now.replace(microsecond=0).isoformat(),
        "last_run": now.replace(microsecond=0).isoformat(),
        "days": days,
        "dates": [d.isoformat() for d in window],
        "n_stations": len(stations),
        "models": [],
        "truth": [],
        "truth_instant": [],
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
        # f000 is not a forecast step: AIWP files carry it, ECMWF and GFS do not, so counting rows
        # would let an AIWP run missing f240 pass as complete (41 rows, 40 expected steps).
        # Same rule, and the same NaN handling, as ``store.existing_inits`` — the two must agree or
        # a run the fetcher considers done would show here as a permanent gap.
        if len(present) and "lead_h" in present.columns:
            step_by_model = {m.model_id: max(int(m.step_h), 1) for m in models}
            min_lead = present["model_id"].map(step_by_model).fillna(1)
            lead_h = pd.to_numeric(present["lead_h"], errors="coerce").fillna(min_lead)
            present = present[lead_h >= min_lead]

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

    # First initialization CastCheck ever held per (model, init hour): nothing before it is a gap.
    first_init: dict[tuple[str, int], pd.Timestamp] = {}
    for (mid, init, _st) in counts:
        key = (mid, int(pd.Timestamp(init).hour))
        cur = first_init.get(key)
        if cur is None or init < cur:
            first_init[key] = init

    upstream_inits = _upstream_inits(models, window[0], window[-1]) if upstream else {}

    for m in models:
        exp = _expected_steps(m)
        for init_hour in m.inits:
            start_init = first_init.get((m.model_id, int(init_hour)))
            up = upstream_inits.get(m.model_id)
            day_rows = []
            for d in window:
                init = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=int(init_hour))
                per_station = [counts.get((m.model_id, init, s), 0) for s in station_ids]
                n_complete = sum(1 for c in per_station if c >= exp)
                n_any = sum(1 for c in per_station if c > 0)
                complete = n_complete == len(station_ids) and len(station_ids) > 0
                # Days that were never expected: before this model entered the record, an
                # initialization the upstream archive did not produce, or a run whose publication
                # deadline has not passed yet (schedule.run_due_at — the same arithmetic the
                # fetcher plans from, so the two can never disagree).
                reason = ""
                due_at = schedule.run_due_at(m, init.to_pydatetime())
                if start_init is not None and init < start_init:
                    reason = "before_start"
                elif not complete and due_at > now:
                    reason = "not_due_yet"
                elif up is not None and init not in up and not complete:
                    reason = "not_produced_upstream"
                expected = reason == ""
                day_rows.append({
                    "date": d.isoformat(),
                    "init_time": init.isoformat(),
                    "due_at": due_at.isoformat(),
                    "complete": bool(complete),
                    "expected": bool(expected),
                    "reason": reason,
                    "stations_complete": n_complete,
                    "stations_any": n_any,
                    "expected_steps": exp,
                    "values": int(sum(per_station)),
                })
                if not complete and expected:
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
            n_expected = sum(1 for r in day_rows if r["expected"])
            report["models"].append({
                "n_not_due_yet": sum(1 for r in day_rows if r["reason"] == "not_due_yet"),
                "model_id": m.model_id,
                "family": m.family,
                "init_hour": int(init_hour),
                "expected_steps": exp,
                "latest_init": latest_init.get(m.model_id),
                "period_start": start_init.date().isoformat() if start_init is not None else None,
                "upstream_known": up is not None,
                "days": day_rows,
                "n_complete": sum(1 for r in day_rows if r["complete"]),
                "n_expected": n_expected,
                "n_not_expected": len(day_rows) - n_expected,
                "n_missing": sum(1 for r in day_rows if not r["complete"] and r["expected"]),
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

    # The newest day whose report *could* exist at all.  Whether it actually should exist yet is a
    # per-station question — the CLI is issued after the station's own local midnight — and is
    # asked below with ``schedule.truth_is_due``.
    truth_deadline = as_of_d - timedelta(days=1)
    for s in stations:
        day_rows = []
        latest_due: date | None = None
        for d in window:
            has = (s.id, d) in final
            due = schedule.truth_is_due(s, d, now)
            if due:
                latest_due = d
            day_rows.append({"date": d.isoformat(), "cli_final": bool(has),
                             "any_source": bool(has or (s.id, d) in partial),
                             "expected": bool(due),
                             "reason": "" if due else "not_due_yet",
                             "due_at": schedule.truth_due_at(s, d).isoformat()})
            if not has and due:
                gap = {"type": "truth", "station_id": s.id, "date": d.isoformat(),
                       "detail": "no first-final CLI with both tmax and tmin"}
                report["gaps"].append(gap)
                if d >= truth_deadline:
                    report["current_gaps"].append(gap)
        report["truth"].append({
            "station_id": s.id, "name": s.name, "cli_pil": s.cli_pil, "days": day_rows,
            "latest_due": latest_due.isoformat() if latest_due is not None else None,
            "n_not_due_yet": sum(1 for r in day_rows if not r["expected"]),
            "n_expected": sum(1 for r in day_rows if r["expected"]),
            "n_missing": sum(1 for r in day_rows if r["expected"] and not r["cli_final"]),
        })

    # ---- truth_instant --------------------------------------------------------------------
    # The observation at 00/06/12/18 UTC is the truth of the headline t2 score, so its coverage
    # belongs on the status page next to the CLI: a station quietly losing one instant a day loses
    # a quarter of its headline sample without anything else turning red.
    inst_counts: dict[tuple[str, date], int] = {}
    if truth_instant is not None and len(truth_instant):
        ti = truth_instant.copy()
        vt = pd.to_datetime(ti["valid_time"], utc=True, errors="coerce")
        keep = vt.notna() & ti["temp_c"].notna() & vt.dt.hour.isin(INSTANT_HOURS)
        for st, d in zip(ti.loc[keep, "station_id"], vt[keep].dt.date):
            inst_counts[(st, d)] = inst_counts.get((st, d), 0) + 1

    # A UTC day's last instant is 18Z, so today is never expected; and, as for the models, a day
    # before this station's own record begins is not a hole — the archive backfill decides that.
    first_instant: dict[str, date] = {}
    for (st, d) in inst_counts:
        cur = first_instant.get(st)
        if cur is None or d < cur:
            first_instant[st] = d
    for st_obj in stations:
        started = first_instant.get(st_obj.id)
        day_rows = []
        for d in window:
            n = inst_counts.get((st_obj.id, d), 0)
            due = schedule.instant_is_due(d, now)
            expected = due and started is not None and d >= started
            day_rows.append({
                "date": d.isoformat(), "n_instants": n,
                "complete": n >= len(INSTANT_HOURS),
                "any_source": n > 0,
                "expected": bool(expected),
                "reason": "" if expected else ("not_due_yet" if not due else "before_start"),
            })
            if expected and n < len(INSTANT_HOURS):
                gap = {"type": "truth_instant", "station_id": st_obj.id, "date": d.isoformat(),
                       "detail": f"{n}/{len(INSTANT_HOURS)} synoptic instants observed"}
                report["gaps"].append(gap)
                if d >= truth_deadline:
                    report["current_gaps"].append(gap)
        report["truth_instant"].append({
            "station_id": st_obj.id, "name": st_obj.name,
            "iem_id": getattr(st_obj, "iem_id", None),
            "period_start": started.isoformat() if started is not None else None,
            "days": day_rows,
            "n_expected": sum(1 for r in day_rows if r["expected"]),
            "n_not_due_yet": sum(1 for r in day_rows if r["reason"] == "not_due_yet"),
            "n_missing": sum(1 for r in day_rows if r["expected"] and not r["complete"]),
        })

    report["n_gaps"] = len(report["gaps"])
    # The per-day grids above already carry the complete picture; the flat list is a convenience,
    # so it is capped (newest first) to keep status.json small. n_gaps stays the true total.
    if len(report["gaps"]) > MAX_LISTED_GAPS:
        report["gaps"] = sorted(report["gaps"], key=lambda g: g["date"],
                                reverse=True)[:MAX_LISTED_GAPS]
        report["gaps_truncated"] = True
    else:
        report["gaps_truncated"] = False
    report["n_current_gaps"] = len(report["current_gaps"])
    report["gaps_today"] = report["current_gaps"]  # alias used by the CLI
    report["ok"] = report["n_current_gaps"] == 0

    # How much of the current day is simply not due yet.  The site says "all due runs present"
    # rather than "all systems operational" while this is non-zero, so a reader at 07 UTC is not
    # told that a 12Z run they cannot possibly have is fine.
    pending = [
        {"type": "model_run", "model_id": m["model_id"], "init_hour": m["init_hour"],
         "date": d["date"], "due_at": d.get("due_at")}
        for m in report["models"] for d in m["days"]
        if d["date"] == as_of_d.isoformat() and d["reason"] == "not_due_yet"
    ] + [
        {"type": "truth", "station_id": t["station_id"], "date": d["date"],
         "due_at": d.get("due_at")}
        for t in report["truth"] for d in t["days"]
        if d["date"] >= truth_deadline.isoformat() and d["reason"] == "not_due_yet"
    ]
    report["pending"] = pending[:MAX_LISTED_GAPS]
    report["n_pending"] = len(pending)

    # GitHub-Status-style headline: share of model-run slots and truth days that are complete.
    # The denominator is the *expected* slots only — days before a model's period_start and
    # initializations the upstream archive never produced are not downtime (review B8).
    slots = [d for m in report["models"] for d in m["days"] if d["expected"]]
    tdays = [d for t in report["truth"] for d in t["days"] if d["expected"]]
    idays = [d for t in report["truth_instant"] for d in t["days"] if d["expected"]]
    report["uptime"] = {
        "model_runs": round(100.0 * sum(1 for d in slots if d["complete"]) / len(slots), 2)
        if slots else None,
        "truth": round(100.0 * sum(1 for d in tdays if d["cli_final"]) / len(tdays), 2)
        if tdays else None,
        "truth_instant": round(100.0 * sum(1 for d in idays if d["complete"]) / len(idays), 2)
        if idays else None,
        "window_days": days,
        "basis": "due slots only: from each model's period_start, upstream-produced "
                 "initializations only, and only what is past its publication deadline",
    }
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
