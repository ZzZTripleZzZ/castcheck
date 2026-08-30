"""ECMWF Open Data adapter: IFS HRES (`oper`) and AIFS Single (`aifs-single`).

Layout (measured 2026-08-30)::

    {base}/{YYYYMMDD}/{HH}z/{ifs|aifs-single}/0p25/oper/{YYYYMMDD}{HH}0000-{step}h-oper-fc.grib2
    {base}/{YYYYMMDD}/{HH}z/{ifs|aifs-single}/0p25/oper/{YYYYMMDD}{HH}0000-{step}h-oper-fc.index

`base` is the live portal ``https://data.ecmwf.int/forecasts`` (~3 days of retention) or the official
AWS mirror ``https://ecmwf-forecasts.s3.amazonaws.com`` (IFS from 2023-01-18, AIFS from 2025-03), which
uses the identical layout. Recent runs try the portal first and fall back to the mirror; older runs go
straight to the mirror.

The ``.index`` file has one JSON object per line with ``param``, ``step``, ``levtype``, ``_offset`` and
``_length``; a ranged GET of ``[_offset, _offset+_length)`` returns exactly one GRIB message (HTTP 206).

Fields collected (METHODOLOGY §2.2, §2.4):

* ``2t`` → stored as ``t2`` (instantaneous, ``bucket_h=0``) at every 6-hourly step out to ``max_h``.
* IFS only: ``mx2t3``/``mn2t3`` at every **3-hourly** step ≤ 144 h (``bucket_h=3``) and
  ``mx2t6``/``mn2t6`` at 6-hourly steps ≥ 150 h (``bucket_h=6``). The 3-hourly steps that are not
  multiples of 6 are required because a ``mx2t3`` message at step *s* covers only ``(s-3, s]`` — without
  them `derive.py` cannot reconstruct a full climatological day of native extremes.

Licence: CC-BY-4.0.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from ..config import ModelSpec
from . import _http
from ._grib import decode_message
from .base import FORECAST_VALUE_COLUMNS, FetchRequest, FetchResult, make_rows, now_utc

log = logging.getLogger(__name__)

PORTAL_BASE = "https://data.ecmwf.int/forecasts"
MIRROR_BASE = "https://ecmwf-forecasts.s3.amazonaws.com"
PORTAL_RETENTION_DAYS = 4
STREAM = "oper"
RESOLUTION = "0p25"

#: GRIB shortName -> (storage variable name, bucket length in hours)
PARAM_TO_VARIABLE: dict[str, tuple[str, int]] = {
    "2t": ("t2", 0),
    "mx2t3": ("mx2t3", 3),
    "mn2t3": ("mn2t3", 3),
    "mx2t6": ("mx2t6", 6),
    "mn2t6": ("mn2t6", 6),
}
NATIVE_EXTREME_SWITCH_H = 144  # ≤144 h: 3-hourly mx2t3/mn2t3; ≥150 h: 6-hourly mx2t6/mn2t6


@dataclass(frozen=True)
class _Task:
    """One GRIB message to fetch: a (step, param) pair of the run."""

    step: int
    param: str

    @property
    def variable(self) -> str:
        return PARAM_TO_VARIABLE[self.param][0]

    @property
    def bucket_h(self) -> int:
        return PARAM_TO_VARIABLE[self.param][1]


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def model_dir(model: ModelSpec) -> str:
    """Path segment for the model: ``ifs`` for HRES, ``aifs-single`` for AIFS."""
    return "ifs" if model.product == "oper" else model.product


def object_url(base: str, model: ModelSpec, init_time: datetime, step: int, suffix: str = "grib2") -> str:
    """Full URL of the GRIB (or ``.index``) object for one step of one run."""
    init = _utc(init_time)
    day = init.strftime("%Y%m%d")
    hh = init.strftime("%H")
    name = f"{day}{hh}0000-{step}h-{STREAM}-fc.{suffix}"
    return f"{base}/{day}/{hh}z/{model_dir(model)}/{RESOLUTION}/{STREAM}/{name}"


def parse_index(text: str) -> list[dict]:
    """Parse an ECMWF ``.index`` file (one JSON object per line); malformed lines are skipped."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.debug("unparseable index line: %.120s", line)
    return out


def find_entry(entries: list[dict], param: str, levtype: str = "sfc") -> dict | None:
    """The index entry for a surface parameter, or None if the run does not carry it."""
    for e in entries:
        if e.get("param") == param and e.get("levtype", "sfc") == levtype and "levelist" not in e:
            return e
    return None


def plan_tasks(model: ModelSpec, variables: tuple[str, ...] | None = None) -> list[_Task]:
    """The (step, param) messages needed for one run of `model`.

    `variables` optionally restricts the plan to a subset of storage variable names.
    """
    tasks: list[_Task] = [_Task(s, "2t") for s in range(model.step_h, model.max_h + 1, model.step_h)]
    natives = set(model.native_extremes)
    if {"mx2t3", "mn2t3"} & natives:
        hi = min(NATIVE_EXTREME_SWITCH_H, model.max_h)
        for s in range(3, hi + 1, 3):
            tasks += [_Task(s, "mx2t3"), _Task(s, "mn2t3")]
    if {"mx2t6", "mn2t6"} & natives and model.max_h > NATIVE_EXTREME_SWITCH_H:
        for s in range(NATIVE_EXTREME_SWITCH_H + 6, model.max_h + 1, 6):
            tasks += [_Task(s, "mx2t6"), _Task(s, "mn2t6")]
    if variables is not None:
        keep = set(variables)
        tasks = [t for t in tasks if t.variable in keep]
    return sorted(tasks, key=lambda t: (t.step, t.param))


def _bases_for(init_time: datetime) -> list[str]:
    """Portal first for runs still inside its ~3-day retention window, mirror first otherwise."""
    age = datetime.now(UTC) - _utc(init_time)
    if age <= timedelta(days=PORTAL_RETENTION_DAYS):
        return [PORTAL_BASE, MIRROR_BASE]
    return [MIRROR_BASE, PORTAL_BASE]


class EcmwfSource:
    """Source adapter for ``source: ecmwf`` models (IFS HRES and AIFS Single)."""

    name = "ecmwf"

    def __init__(self, workers: int = _http.MAX_WORKERS, batch: int = 16) -> None:
        self.workers = max(1, workers)
        self.batch = max(1, batch)

    # ------------------------------------------------------------------ inits

    def available_inits(self, model: ModelSpec, start: date, end: date) -> list[datetime]:
        """Initialisations in ``[start, end]`` whose final needed step is published.

        Probes the ``.index`` of the longest step we use (a run whose last step exists is complete for
        our purposes), portal or mirror.
        """
        candidates: list[datetime] = []
        d = start
        while d <= end:
            for hh in model.inits:
                candidates.append(datetime(d.year, d.month, d.day, hh, tzinfo=UTC))
            d += timedelta(days=1)

        def probe(init: datetime) -> datetime | None:
            for base in _bases_for(init):
                url = object_url(base, model, init, model.max_h, "index")
                if _http.fetch(url, head=True, retries=2).ok:
                    return init
            return None

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            found = list(pool.map(probe, candidates))
        return [i for i in found if i is not None]

    # ------------------------------------------------------------------ fetch

    def fetch_run(self, req: FetchRequest) -> FetchResult:
        """Extract all station values for one run. Never raises: failures become missing rows."""
        model, init = req.model, _utc(req.init_time)
        stations = req.stations
        tasks = plan_tasks(model, req.variables)
        notes: list[str] = []
        fetched_at = now_utc()
        state = {"model_version": "unknown"}

        def emit(task: _Task, values: dict, url: str, reason: str) -> pd.DataFrame:
            return make_rows(
                model=model,
                model_version=state["model_version"],
                init_time=init,
                valid_time=init + timedelta(hours=task.step),
                lead_h=task.step,
                variable=task.variable,
                bucket_h=task.bucket_h,
                source_url=url,
                values=values,
                stations=stations,
                missing_reason=reason,
                fetched_at=fetched_at,
            )

        # --- 1. is the run there at all? -------------------------------------
        steps = sorted({t.step for t in tasks})
        probe_url, probe_reason = "", ""
        base_order = _bases_for(init)
        for base in base_order:
            url = object_url(base, model, init, steps[0], "index")
            res = _http.fetch(url)
            if not probe_reason:  # report the primary archive's verdict, not the fallback's
                probe_url, probe_reason = url, res.reason
            if res.ok:
                base_order = [base] + [b for b in base_order if b != base]
                break
        else:
            notes.append(f"run not found ({probe_reason}) at {probe_url}")
            log.warning("ECMWF run %s %s unavailable: %s", model.model_id, init.isoformat(), probe_reason)
            frames = [
                emit(t, {}, object_url(base_order[0], model, init, t.step), probe_reason or "no_file")
                for t in tasks
            ]
            return FetchResult(_concat(frames), notes)

        # --- 2. indices for every needed step --------------------------------
        def load_index(step: int) -> tuple[int, list[dict], str, str]:
            reason = ""
            for base in base_order:
                res = _http.fetch(object_url(base, model, init, step, "index"))
                if res.ok:
                    return step, parse_index(res.text), object_url(base, model, init, step), ""
                reason = reason or res.reason
            return step, [], object_url(base_order[0], model, init, step), reason or "no_file"

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            index_by_step = {s: (e, u, r) for s, e, u, r in pool.map(load_index, steps)}

        # --- 3. byte ranges ---------------------------------------------------
        ranged: list[tuple[_Task, str, tuple[int, int]]] = []
        frames: list[pd.DataFrame] = []
        for task in tasks:
            entries, grib_url, reason = index_by_step[task.step]
            if reason:
                frames.append(emit(task, {}, grib_url, reason))
                continue
            entry = find_entry(entries, task.param)
            if entry is None:
                frames.append(emit(task, {}, grib_url, "no_field"))
                continue
            off = int(entry["_offset"])
            ranged.append((task, grib_url, (off, off + int(entry["_length"]) - 1)))

        # --- 4. download (threaded) + decode (serial; eccodes is not thread-safe)
        for chunk in _chunks(ranged, self.batch):
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                bodies = list(pool.map(lambda a: _http.fetch(a[1], byte_range=a[2]), chunk))
            for (task, grib_url, _), res in zip(chunk, bodies):
                if not res.ok:
                    frames.append(emit(task, {}, grib_url, res.reason or "no_file"))
                    continue
                dec = decode_message(res.content or b"", stations)
                if dec.error:
                    frames.append(emit(task, {}, grib_url, dec.error))
                    continue
                if state["model_version"] == "unknown" and dec.generating_process is not None:
                    state["model_version"] = _model_version(model, dec.generating_process)
                frames.append(emit(task, dec.values, grib_url, "fill_value"))

        rows = _concat(frames)
        if state["model_version"] == "unknown":
            notes.append("model_version unknown (no generatingProcessIdentifier decoded)")
        else:
            rows["model_version"] = rows["model_version"].replace("unknown", state["model_version"])
        return FetchResult(rows, notes)


def _model_version(model: ModelSpec, gpid: int) -> str:
    """Stable per-cycle identifier, e.g. ``ifs-gpid161`` / ``aifs-single-gpid...``."""
    return f"{model_dir(model)}-gpid{gpid}"


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=FORECAST_VALUE_COLUMNS)
    return pd.concat(frames, ignore_index=True)
