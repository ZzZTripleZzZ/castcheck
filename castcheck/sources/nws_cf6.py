"""NWS Preliminary Monthly Climate Data (CF6 / WS Form F-6) parsing (METHODOLOGY §3).

CF6 is a whole-month table re-issued every morning (~09:10 UTC) with one row per day. It is used for
month-end reconciliation and to fill CLI gaps. Like CLI, ``api.weather.gov`` keeps only about a
week, so older months come from the IEM AFOS archive under the pil ``CF6{LOC}``.

Table layout (fixed columns, whitespace separated)::

    DY MAX MIN AVG DEP HDD CDD  WTR  SNW DPTH SPD SPD DIR MIN PSBL S-S WX    SPD DR
     1  86  72  79   1   0  14 0.00  0.0    0  3.6 10 220   M    M   0        20 170
"""

from __future__ import annotations

import calendar
import contextlib
import logging
import re
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from ..config import Station
from .nws_cli import _WMO_RE, MONTHS, NWS_API, _get, iem_cli_history

log = logging.getLogger(__name__)

CF6_COLUMNS = [
    "climo_date", "tmax_f", "tmin_f", "tavg_f", "dep_f", "hdd", "cdd", "precip_in", "snow_in",
    "snow_depth_in", "wind_avg_mph", "wind_max_mph", "wind_dir", "product_id", "issuance_time",
]

_HEADER_RE = re.compile(r"MONTH:\s*(?P<mon>[A-Z]+)\s.*?YEAR:\s*(?P<year>\d{4})", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"^\s{0,2}(?P<dy>\d{1,2})\s+(?P<rest>[-\dMT. ]+.*)$")


def _num(token: str) -> float | None:
    token = token.strip()
    if token in ("", "M", "MM", "*"):
        return None
    if token == "T":  # trace
        return 0.0
    try:
        return float(token)
    except ValueError:
        return None


def _int(token: str) -> int | None:
    v = _num(token)
    return None if v is None else round(v)


def _col(cols: list[str], i: int) -> str:
    return cols[i] if i < len(cols) else ""


def _issuance_from_wmo(text: str, year: int, month: int, last_data_day: int) -> datetime | None:
    """CF6 carries no local-time line, so rebuild the UTC issuance from the WMO ``DDHHMM`` header.

    The product for month *M* is issued either during *M* (partial month) or early in *M+1* (final),
    so both candidate dates are tried and the earliest one at or after the last data day wins.
    """
    wmo = _WMO_RE.search(text)
    if not wmo:
        return None
    dd, hh, mi = (int(wmo.group("ddhhmm")[i: i + 2]) for i in (0, 2, 4))
    cands: list[datetime] = []
    for y, m in ((year, month), (year + (month == 12), month % 12 + 1)):
        if dd <= calendar.monthrange(y, m)[1]:
            with contextlib.suppress(ValueError):  # e.g. 29 February in a non-leap year
                cands.append(datetime(y, m, dd, hh, mi, tzinfo=UTC))
    if not cands:
        return None
    floor = datetime(year, month, min(last_data_day, calendar.monthrange(year, month)[1]),
                     tzinfo=UTC)
    later = [c for c in cands if c >= floor]
    return min(later) if later else max(cands)


def parse_cf6(text: str, issuance_time: datetime | None = None) -> pd.DataFrame:
    """Parse one CF6 product into a daily table. Returns an empty frame if it is not a CF6.

    ``issuance_time`` overrides the timestamp rebuilt from the WMO header (pass the authoritative
    ``issuanceTime`` when the product came from api.weather.gov).
    """
    if "F-6" not in text and "PRELIMINARY LOCAL CLIMATOLOGICAL DATA" not in text.upper():
        return pd.DataFrame(columns=CF6_COLUMNS)
    head = _HEADER_RE.search(text)
    if not head:
        return pd.DataFrame(columns=CF6_COLUMNS)
    month = MONTHS.get(head.group("mon").upper())
    year = int(head.group("year"))
    if month is None:
        return pd.DataFrame(columns=CF6_COLUMNS)
    ndays = calendar.monthrange(year, month)[1]

    # Only the block between the "DY MAX MIN" header and the "SM" summary line holds day rows.
    start = text.find("DY MAX MIN")
    body = text[start:] if start >= 0 else text
    end = re.search(r"^\s*SM\b", body, re.MULTILINE)
    if end:
        body = body[: end.start()]

    recs = []
    seen: set[int] = set()
    for line in body.splitlines():
        if line.startswith("DY") or set(line.strip()) <= {"=", "-"}:
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        dy = int(m.group("dy"))
        if not 1 <= dy <= ndays or dy in seen:
            continue
        cols = line.split()
        if len(cols) < 4:
            continue
        seen.add(dy)
        recs.append(
            {
                "climo_date": date(year, month, dy),
                "tmax_f": _int(_col(cols, 1)),
                "tmin_f": _int(_col(cols, 2)),
                "tavg_f": _int(_col(cols, 3)),
                "dep_f": _int(_col(cols, 4)),
                "hdd": _int(_col(cols, 5)),
                "cdd": _int(_col(cols, 6)),
                "precip_in": _num(_col(cols, 7)),
                "snow_in": _num(_col(cols, 8)),
                "snow_depth_in": _num(_col(cols, 9)),
                "wind_avg_mph": _num(_col(cols, 10)),
                "wind_max_mph": _num(_col(cols, 11)),
                "wind_dir": _int(_col(cols, 12)),
            }
        )
    df = pd.DataFrame.from_records(recs, columns=CF6_COLUMNS)
    if df.empty:
        return df
    df = df.sort_values("climo_date").reset_index(drop=True)

    issuance = issuance_time or _issuance_from_wmo(text, year, month, int(seen and max(seen) or 1))
    wmo = _WMO_RE.search(text)
    pil_m = re.search(r"^(CF6[A-Z0-9]{2,4})\s*$", text, re.MULTILINE)
    df["issuance_time"] = pd.Timestamp(issuance) if issuance is not None else pd.NaT
    df["product_id"] = (
        f"{issuance:%Y%m%d%H%M}-{wmo.group('cccc')}-{wmo.group('ttaaii')}-{pil_m.group(1)}"
        if (wmo and pil_m and issuance is not None) else ""
    )
    return df


def _cf6_pil(station: Station) -> str:
    return "CF6" + station.cli_location


def fetch_cf6(station: Station, year: int, month: int) -> pd.DataFrame:
    """Daily CF6 table for one station-month (columns: ``climo_date``, ``tmax_f``, ``tmin_f``, …).

    Tries the live api.weather.gov product list first (only ~7 days of retention, so this works for
    the current and previous month) and falls back to the IEM AFOS archive. Returns an empty frame
    with the right columns when nothing is available.
    """
    df = _fetch_cf6_api(station, year, month)
    if not df.empty and df["tmax_f"].notna().any():
        return df
    return _fetch_cf6_iem(station, year, month)


def _fetch_cf6_api(station: Station, year: int, month: int) -> pd.DataFrame:
    url = f"{NWS_API}/products/types/CF6/locations/{station.cli_location}"
    try:
        r = _get(url)
    except RuntimeError:
        return pd.DataFrame(columns=CF6_COLUMNS)
    if r.status_code != 200:
        return pd.DataFrame(columns=CF6_COLUMNS)
    graph = r.json().get("@graph") or []
    best = pd.DataFrame(columns=CF6_COLUMNS)
    for p in sorted(graph, key=lambda x: x["issuanceTime"], reverse=True):
        try:
            text = _get(f"{NWS_API}/products/{p['id']}").json()["productText"]
        except Exception as exc:  # noqa: BLE001 - one bad product must not lose the month
            log.warning("CF6 product %s unreadable: %s: %s", p["id"], type(exc).__name__, exc)
            continue
        df = parse_cf6(text, issuance_time=datetime.fromisoformat(p["issuanceTime"]).astimezone(UTC))
        if df.empty:
            continue
        if df["climo_date"].iloc[0].year == year and df["climo_date"].iloc[0].month == month:
            if len(df) > len(best):
                best = df
            # the newest matching issuance is the most complete; stop once we have a full month
            if len(best) >= calendar.monthrange(year, month)[1]:
                break
    return best


def _fetch_cf6_iem(station: Station, year: int, month: int) -> pd.DataFrame:
    """Latest archived CF6 covering ``year``/``month`` (issued during or just after that month)."""
    last = calendar.monthrange(year, month)[1]
    start = date(year, month, max(1, last - 2))
    end = date(year, month, last) + timedelta(days=6)
    texts = iem_cli_history(_cf6_pil(station), start, end)
    best = pd.DataFrame(columns=CF6_COLUMNS)
    for text in texts:
        df = parse_cf6(text)
        if df.empty:
            continue
        if df["climo_date"].iloc[0].year != year or df["climo_date"].iloc[0].month != month:
            continue
        if len(df) >= len(best):
            best = df
    return best
