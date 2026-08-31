"""NCEP GFS 0.25° adapter (AWS open-data mirror, NOMADS fallback).

Layout (measured 2026-08-30)::

    https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{YYYYMMDD}/{HH}/atmos/gfs.t{HH}z.pgrb2.0p25.f{FFF}
    …same key + ".idx"

The ``.idx`` is wgrib2 inventory text, one record per line::

    586:423312699:d=2026083000:TMAX:2 m above ground:0-6 hour max fcst:

Fields are ``n:offset:date:VAR:level:description:``. A record ends where the *next* record starts, so a
byte range is ``[offset_i, offset_{i+1} - 1]`` (open-ended for the last record).

Fields collected (METHODOLOGY §2.2, §2.4), at 6-hourly steps out to ``max_h``:

* ``TMP:2 m above ground:{h} hour fcst`` → ``t2`` (instantaneous, ``bucket_h=0``)
* ``TMAX:2 m above ground:{h-6}-{h} hour max fcst`` → ``tmax6`` (``bucket_h=6``)
* ``TMIN:2 m above ground:{h-6}-{h} hour min fcst`` → ``tmin6`` (``bucket_h=6``)

Longitudes are 0..359.75 (handled inside `grid.py`). The AWS archive starts 2021-01-01; the data are
public domain. `model_version` is ``gfs-0p25`` — the inventory carries no cycle identifier.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from ..config import ModelSpec
from . import _http
from ._grib import decode_message
from .base import FORECAST_VALUE_COLUMNS, FetchRequest, FetchResult, make_rows, now_utc

log = logging.getLogger(__name__)

AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
MODEL_VERSION = "gfs-0p25"
LEVEL_2M = "2 m above ground"
BUCKET_H = 6

#: Retries for a single ``.idx`` request, and the per-run budget of index failures before the rest of
#: the steps are marked missing without a request (see the note in `ecmwf.py`).
INDEX_RETRIES = 3
MAX_INDEX_FAILURES = 5


class _Counter:
    """A thread-safe counter (the index loaders run in a pool and share one failure budget)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0

    def increment(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    @property
    def value(self) -> int:
        with self._lock:
            return self._n


@dataclass(frozen=True)
class _Task:
    """One inventory record to fetch: variable name plus the ``.idx`` fields identifying it."""

    step: int
    variable: str
    bucket_h: int
    var_key: str  # wgrib2 VAR field
    desc: str  # wgrib2 description field


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def object_url(base: str, init_time: datetime, step: int, suffix: str = "") -> str:
    """Full URL of one GFS pgrb2 0.25° step (append ``.idx`` for the inventory)."""
    init = _utc(init_time)
    day = init.strftime("%Y%m%d")
    hh = init.strftime("%H")
    return f"{base}/gfs.{day}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{step:03d}{suffix}"


def parse_idx(text: str) -> list[dict]:
    """Parse a wgrib2 ``.idx`` inventory into records with byte ranges.

    Each record gets ``offset`` and ``end`` (inclusive, ``None`` for the final record).
    """
    recs: list[dict] = []
    for line in text.splitlines():
        parts = line.strip().split(":")
        if len(parts) < 6 or not parts[1].isdigit():
            continue
        recs.append(
            {
                "n": int(parts[0]),
                "offset": int(parts[1]),
                "date": parts[2],
                "var": parts[3],
                "level": parts[4],
                "desc": parts[5],
                "end": None,
            }
        )
    for i in range(len(recs) - 1):
        recs[i]["end"] = recs[i + 1]["offset"] - 1
    return recs


def find_record(recs: list[dict], var: str, level: str, desc: str) -> dict | None:
    for r in recs:
        if r["var"] == var and r["level"] == level and r["desc"] == desc:
            return r
    return None


def plan_tasks(model: ModelSpec, variables: tuple[str, ...] | None = None) -> list[_Task]:
    """The inventory records needed for one run: t2 plus the 6-hour TMAX/TMIN buckets."""
    tasks: list[_Task] = []
    step_h = max(model.step_h, BUCKET_H)
    for step in range(step_h, model.max_h + 1, step_h):
        tasks.append(_Task(step, "t2", 0, "TMP", f"{step} hour fcst"))
        if model.native_extremes:
            a = step - BUCKET_H
            tasks.append(_Task(step, "tmax6", BUCKET_H, "TMAX", f"{a}-{step} hour max fcst"))
            tasks.append(_Task(step, "tmin6", BUCKET_H, "TMIN", f"{a}-{step} hour min fcst"))
    if variables is not None:
        keep = set(variables)
        tasks = [t for t in tasks if t.variable in keep]
    return tasks


class GfsSource:
    """Source adapter for ``source: gfs`` models."""

    name = "gfs"

    def __init__(self, workers: int = _http.MAX_WORKERS, batch: int = 16) -> None:
        self.workers = max(1, workers)
        self.batch = max(1, batch)

    # ------------------------------------------------------------------ inits

    def available_inits(self, model: ModelSpec, start: date, end: date) -> list[datetime]:
        """Initialisations in ``[start, end]`` whose final needed step is published on AWS/NOMADS."""
        candidates: list[datetime] = []
        d = start
        while d <= end:
            for hh in model.inits:
                candidates.append(datetime(d.year, d.month, d.day, hh, tzinfo=UTC))
            d += timedelta(days=1)

        def probe(init: datetime) -> datetime | None:
            for base in (AWS_BASE, NOMADS_BASE):
                if _http.fetch(object_url(base, init, model.max_h, ".idx"), head=True, retries=2).ok:
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

        def emit(task: _Task, values: dict, url: str, reason: str) -> pd.DataFrame:
            return make_rows(
                model=model,
                model_version=MODEL_VERSION,
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

        steps = sorted({t.step for t in tasks})
        base_order = [AWS_BASE, NOMADS_BASE]
        probe_reason = ""
        for base in base_order:
            res = _http.fetch(object_url(base, init, steps[0], ".idx"))
            if not probe_reason:  # report the primary archive's verdict, not the fallback's
                probe_reason = res.reason
            if res.ok:
                base_order = [base] + [b for b in base_order if b != base]
                break
        else:
            notes.append(f"run not found ({probe_reason}) at {object_url(AWS_BASE, init, steps[0])}")
            log.warning("GFS run %s unavailable: %s", init.isoformat(), probe_reason)
            frames = [
                emit(t, {}, object_url(AWS_BASE, init, t.step), probe_reason or "no_file") for t in tasks
            ]
            return FetchResult(_concat(frames), notes)

        failures = _Counter()

        def load_idx(step: int) -> tuple[int, list[dict], str, str]:
            fallback_url = object_url(base_order[0], init, step)
            if failures.value >= MAX_INDEX_FAILURES:
                return step, [], fallback_url, "index_unavailable"
            reason = ""
            for base in base_order:
                res = _http.fetch(object_url(base, init, step, ".idx"), retries=INDEX_RETRIES)
                if res.ok:
                    return step, parse_idx(res.text), object_url(base, init, step), ""
                reason = reason or res.reason
            failures.increment()
            return step, [], fallback_url, reason or "no_file"

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            idx_by_step = {s: (r, u, why) for s, r, u, why in pool.map(load_idx, steps)}
        if failures.value:
            note = f"{failures.value} of {len(steps)} step inventories unavailable"
            if failures.value >= MAX_INDEX_FAILURES:
                note += f" (gave up after {MAX_INDEX_FAILURES})"
            notes.append(note)
            log.warning("GFS %s: %s", init.isoformat(), note)

        ranged: list[tuple[_Task, str, tuple[int, int | None]]] = []
        frames: list[pd.DataFrame] = []
        for task in tasks:
            recs, grib_url, reason = idx_by_step[task.step]
            if reason:
                frames.append(emit(task, {}, grib_url, reason))
                continue
            rec = find_record(recs, task.var_key, LEVEL_2M, task.desc)
            if rec is None:
                frames.append(emit(task, {}, grib_url, "no_field"))
                continue
            ranged.append((task, grib_url, (rec["offset"], rec["end"])))

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
                frames.append(emit(task, dec.values, grib_url, "fill_value"))

        return FetchResult(_concat(frames), notes)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=FORECAST_VALUE_COLUMNS)
    return pd.concat(frames, ignore_index=True)
