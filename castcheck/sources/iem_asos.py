"""Hourly METAR air temperature from the IEM ASOS archive (DESIGN §10.1).

The Iowa Environmental Mesonet mirrors the raw ASOS/AWOS METAR stream back to the 1930s for US
airports and serves it as CSV from a single CGI endpoint. It is the archive behind
``truth_instant``: the observed 2 m temperature at the four synoptic instants the models are
sampled at, which is what makes the headline metric a like-for-like comparison instead of a
comparison against a differently-defined daily extreme (external review A2).

What is requested
-----------------
``report_type=3`` is IEM's *Routine / Once Hourly* class — the scheduled METAR near the end of each
hour (:51–:56 at most US sites). SPECIs (``report_type=4``) and the MADIS 5-minute stream
(``report_type=1``) are deliberately excluded: SPECIs fire on weather changes and would bias the
sample towards eventful minutes, and the 5-minute values are rounded to whole °C, which is a
different rounding from the routine report and would mix two quantisations in one series.

``data=tmpf`` is the temperature as the archive holds it: whole degrees Fahrenheit, the resolution
the METAR body reports. It is converted to °C here and stored as ``float32``. The tenths-°C ``Tddd``
remark group is finer but is absent from a minority of reports, so using it would make the series'
resolution depend on the station and the year.

Station ids
-----------
IEM identifies CONUS ASOS sites by the three-character id without the leading ``K`` (``KORD`` →
``ORD``). The endpoint accepts the four-character ICAO too and normalises it, but the id that is
*known* to work for each station is frozen in ``config/stations.yaml`` as ``iem_id`` rather than
derived at run time, so a station whose id does not follow the rule cannot fail silently.

Rate
----
IEM is a volunteer-run academic archive, not an object store. Requests are chunked by month and
paced at :data:`MIN_INTERVAL_S` seconds per request through the shared per-host limiter in
:mod:`castcheck.sources._http`. Asking for more than that earns a plain-text ``Too many requests``
body served with **HTTP 200**, which no status-code check can catch — see :func:`looks_like_csv`.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from ..config import Station
from . import _http

log = logging.getLogger(__name__)

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

#: IEM ``report_type`` codes (see the checkboxes on /request/download.phtml).
REPORT_TYPE_HFMETAR = 1  # MADIS 5-minute ASOS
REPORT_TYPE_ROUTINE = 3  # scheduled hourly METAR — the only one used here
REPORT_TYPE_SPECIAL = 4  # SPECI

#: seconds between two requests to the IEM host, minimum
MIN_INTERVAL_S = 1.2

#: attempts per station-month when the archive answers with its throttle notice
THROTTLE_RETRIES = 5
#: first wait after a throttle notice, seconds; doubled each attempt
THROTTLE_BACKOFF_S = 5.0

ASOS_COLUMNS = ["obs_time", "temp_c", "report_type"]

#: first line of a well-formed response; anything else is an error page or the throttle notice
_CSV_HEADER = "station,valid"

#: values IEM uses for "no data" in the CSV (``missing=M``, ``trace=T``)
_MISSING = frozenset({"", "M", "m", "null", "NULL", "None"})


def iem_id(station: Station) -> str:
    """The archive's id for a station: the frozen ``iem_id``, else the ICAO without a leading ``K``."""
    fixed = getattr(station, "iem_id", None)
    if fixed:
        return str(fixed)
    sid = station.id
    return sid[1:] if len(sid) == 4 and sid.startswith("K") else sid


def f_to_c(f: float) -> float:
    return (float(f) - 32.0) * 5.0 / 9.0


def _empty() -> pd.DataFrame:
    df = pd.DataFrame(columns=ASOS_COLUMNS)
    df["obs_time"] = pd.to_datetime(df["obs_time"], utc=True)
    df["temp_c"] = df["temp_c"].astype("float32")
    df["report_type"] = df["report_type"].astype("int8")
    return df


def _month_chunks(start: datetime, end: datetime) -> list[tuple[date, date]]:
    """``[start, end]`` split at month boundaries, as inclusive date pairs.

    The endpoint's ``day2`` bound is exclusive of the following day but inclusive of ``day2``'s own
    reports, so each chunk is asked for as a whole closed date interval and the caller trims to the
    real timestamp window afterwards.
    """
    out: list[tuple[date, date]] = []
    d0, d1 = start.date(), end.date()
    cur = d0
    while cur <= d1:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cur, min(d1, nxt - timedelta(days=1))))
        cur = nxt
    return out


def _url(sid: str, first: date, last: date, report_type: int) -> str:
    """The archive URL for one station-month. ``day2`` is bumped by a day: IEM's end bound is the
    *start* of that day, so asking for the month's last day alone would drop its reports."""
    stop = last + timedelta(days=1)
    q = (
        f"station={sid}&data=tmpf"
        f"&year1={first.year}&month1={first.month}&day1={first.day}"
        f"&year2={stop.year}&month2={stop.month}&day2={stop.day}"
        "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T"
        f"&report_type={report_type}"
    )
    return f"{IEM_ASOS}?{q}"


def looks_like_csv(text: str) -> bool:
    """Whether a response body is the CSV we asked for rather than a message about it.

    When the archive is being asked for too much it answers ``Too many requests from your IP
    address, slow down.`` **with HTTP 200**, so nothing below the transport layer can tell that the
    request failed. Parsing that body yields zero rows, which is indistinguishable from a station
    that genuinely reported nothing — a silent hole in a backfill, which is the one failure mode
    this table cannot tolerate. The body is therefore checked for the header line it must start
    with.
    """
    return text.lstrip().lower().startswith(_CSV_HEADER)


def _fetch_month(sid: str, first: date, last: date, report_type: int) -> str | None:
    """One station-month of CSV, retried through the archive's throttle notice. ``None`` if lost."""
    url = _url(sid, first, last, report_type)
    wait = THROTTLE_BACKOFF_S
    for attempt in range(THROTTLE_RETRIES):
        res = _http.fetch(url, timeout=120)
        if not res.ok:
            log.warning("IEM ASOS %s %s..%s: %s", sid, first, last, res.reason)
            return None
        if looks_like_csv(res.text):
            return res.text
        note = " ".join(res.text.split())[:80]
        if attempt == THROTTLE_RETRIES - 1:
            log.warning("IEM ASOS %s %s..%s: gave up after %d throttled replies (%s)",
                        sid, first, last, THROTTLE_RETRIES, note)
            return None
        log.info("IEM ASOS %s %s..%s: %s — waiting %.0fs", sid, first, last, note, wait)
        time.sleep(wait)
        wait *= 2
    return None


def parse_asos_csv(text: str, report_type: int = REPORT_TYPE_ROUTINE) -> pd.DataFrame:
    """Parse the ``format=onlycomma`` CSV into ``obs_time`` (UTC), ``temp_c``, ``report_type``.

    Rows whose timestamp will not parse are dropped (the archive occasionally emits a truncated
    trailing line); a missing temperature becomes ``NaN`` and keeps its row, because "the station
    reported at this minute but without a temperature" is different from "the station was silent",
    and :mod:`castcheck.truth_instant` distinguishes the two.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    recs = []
    for r in rows:
        raw_t = (r.get("valid") or "").strip()
        if not raw_t:
            continue
        ts = pd.to_datetime(raw_t, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        raw_v = (r.get("tmpf") or "").strip()
        try:
            temp = float("nan") if raw_v in _MISSING else f_to_c(float(raw_v))
        except ValueError:
            temp = float("nan")
        recs.append({"obs_time": ts, "temp_c": temp, "report_type": report_type})
    if not recs:
        return _empty()
    df = pd.DataFrame.from_records(recs, columns=ASOS_COLUMNS)
    df["temp_c"] = df["temp_c"].astype("float32")
    df["report_type"] = df["report_type"].astype("int8")
    return (df.sort_values("obs_time").drop_duplicates("obs_time", keep="last").reset_index(drop=True))


def fetch_asos(
    station: Station, start: datetime, end: datetime, *, report_type: int = REPORT_TYPE_ROUTINE,
) -> pd.DataFrame:
    """Routine hourly METAR temperatures for ``[start, end]`` (UTC, inclusive).

    Returns ``obs_time`` (UTC), ``temp_c`` (float32, ``NaN`` when the report carried no
    temperature) and ``report_type``. A month whose request fails is logged and skipped rather than
    raised: a gap in the archive must show up as a ``no_report`` flag on the affected instants, not
    as a lost backfill.
    """
    start, end = _as_utc(start), _as_utc(end)
    if end < start:
        return _empty()
    _http.set_min_interval(IEM_ASOS, MIN_INTERVAL_S)
    sid = iem_id(station)
    frames = []
    for first, last in _month_chunks(start, end):
        text = _fetch_month(sid, first, last, report_type)
        if text is None:
            continue
        frames.append(parse_asos_csv(text, report_type))
    if not frames:
        return _empty()
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["obs_time"] >= start) & (df["obs_time"] <= end)]
    return df.sort_values("obs_time").drop_duplicates("obs_time", keep="last").reset_index(drop=True)


def _as_utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_convert(UTC) if ts.tzinfo else ts.tz_localize(UTC)


__all__ = [
    "ASOS_COLUMNS",
    "IEM_ASOS",
    "REPORT_TYPE_ROUTINE",
    "f_to_c",
    "looks_like_csv",
    "fetch_asos",
    "iem_id",
    "parse_asos_csv",
]
