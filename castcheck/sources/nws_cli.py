"""NWS Daily Climate Report (CLI) discovery and parsing (METHODOLOGY §3, DESIGN §4).

Two sources are used:

* ``https://api.weather.gov/products/types/CLI/locations/{LOC}`` — live, ~6-7 days of retention.
  ``LOC`` is the AFOS pil without the leading ``CLI`` (``CLINYC`` → ``NYC``). A real ``User-Agent``
  is mandatory.
* ``https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py`` — the Iowa Environmental Mesonet
  AFOS archive, used for history. Products come back concatenated, each wrapped in SOH/ETX
  (``\\x01`` … ``\\x03``); ``limit`` defaults to 1 upstream so it must always be passed.

Truth policy (METHODOLOGY §3): the value for a climatological day is the ``YESTERDAY`` block of the
*first* CLI issued after local midnight. Same-day ``TODAY``/``VALID AS OF`` reports are intermediate
and never used. Later corrections are recorded but do not replace the first final.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, timezone

import requests

from ..climo_day import day_bounds_utc
from ..config import USER_AGENT, Station

NWS_API = "https://api.weather.gov"
IEM_AFOS = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"

MONTHS = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6, "JULY": 7,
    "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9,
    "OCT": 10, "NOV": 11, "DEC": 12,
}

# UTC offsets of the time-zone abbreviations that appear in CLI issuance lines.
TZ_ABBR = {
    "UTC": 0, "GMT": 0, "AST": -4, "ADT": -3, "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8, "HST": -10, "HDT": -9,
    "SST": -11, "CHST": 10,
}

_TITLE_RE = re.compile(
    r"\.\.\.[ \t]*(?:THE[ \t]+)?(?P<name>[^\n.][^\n]*?)[ \t]+CLIMATE[ \t]+SUMMARY[ \t]+FOR[ \t]+"
    r"(?P<mon>[A-Z]+)[ \t]+(?P<day>\d{1,2})(?:[ \t]+(?P<year>\d{4}))?",
    re.IGNORECASE,
)
_ISSUE_RE = re.compile(
    r"^\s*(?P<hm>\d{3,4})\s+(?P<ampm>AM|PM)\s+(?P<tz>[A-Z]{2,4})\s+[A-Z]{3}\s+"
    r"(?P<mon>[A-Z]{3})\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})\s*$",
    re.MULTILINE,
)
_WMO_RE = re.compile(r"^(?P<ttaaii>[A-Z]{4}\d{2})\s+(?P<cccc>[A-Z]{4})\s+(?P<ddhhmm>\d{6})(?P<suffix>[^\n]*)$",
                     re.MULTILINE)
_BLOCK_RE = re.compile(r"^[ \t]*(YESTERDAY|TODAY)[ \t]*$", re.MULTILINE)
_EXTREME_RE = re.compile(r"^\s*(?P<what>MAXIMUM|MINIMUM)\s+(?P<val>MM|M|-?\d+)(?P<flag>[A-Z]*)(?=\s|$)")
_TIME_RE = re.compile(r"^\s*(?P<h>\d{1,2}):?(?P<m>\d{2})\s*(?P<ampm>AM|PM)")
_SECTION_END_RE = re.compile(r"^\s*(PRECIPITATION|SNOWFALL|DEGREE DAYS|WIND|RELATIVE HUMIDITY|SKY COVER)",
                             re.MULTILINE)

_SESSION: requests.Session | None = None


def session() -> requests.Session:
    """Process-wide requests session carrying the mandatory User-Agent."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json, */*"})
        _SESSION = s
    return _SESSION


def _get(url: str, *, params: dict | None = None, timeout: int = 60, retries: int = 3) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session().get(url, params=params, timeout=timeout)
            if r.status_code >= 500 and attempt < retries - 1:
                last = RuntimeError(f"HTTP {r.status_code}")
                continue
            return r
        except requests.RequestException as exc:  # pragma: no cover - network flake
            last = exc
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


# --------------------------------------------------------------------------- discovery


def list_cli_products(station: Station, limit: int = 50) -> list[dict]:
    """List recent CLI products for a station, newest first.

    Returns dicts with ``product_id``, ``issuance_time`` (aware UTC ``datetime``), ``office`` and
    ``pil``. An empty list is returned for a 404 (unknown location) rather than raising.
    """
    url = f"{NWS_API}/products/types/CLI/locations/{station.cli_location}"
    r = _get(url)  # the endpoint rejects a ?limit= parameter; it already returns only ~7 days
    if r.status_code == 404:
        return []
    r.raise_for_status()
    graph = r.json().get("@graph") or []
    out = []
    for p in graph[:limit]:
        out.append(
            {
                "product_id": p["id"],
                "issuance_time": datetime.fromisoformat(p["issuanceTime"]).astimezone(UTC),
                "office": p.get("issuingOffice", ""),
                "pil": station.cli_pil,
            }
        )
    out.sort(key=lambda d: d["issuance_time"], reverse=True)
    return out


def get_product_text(product_id: str) -> str:
    """Fetch the raw text of one api.weather.gov product."""
    r = _get(f"{NWS_API}/products/{product_id}")
    r.raise_for_status()
    return r.json()["productText"]


def iem_cli_history(pil: str, start: date, end: date) -> list[str]:
    """All archived AFOS products for ``pil`` issued in ``[start, end]`` (inclusive), oldest first.

    IEM's ``edate`` is exclusive and its ``limit`` defaults to 1, so both are adjusted here. The
    response concatenates products delimited by ``\\x01``/``\\x03``.
    """
    ndays = max((end - start).days, 0) + 1
    params = {
        "pil": pil,
        "sdate": start.isoformat(),
        "edate": (end + timedelta(days=1)).isoformat(),  # edate is exclusive upstream
        "fmt": "text",
        "limit": max(50, ndays * 8),
    }
    r = _get(IEM_AFOS, params=params, timeout=180)
    if r.status_code != 200:
        return []
    texts = [p.replace("\x03", "").strip("\n") for p in r.text.split("\x01")]
    texts = [p for p in texts if p.strip()]
    texts.sort(key=lambda t: (_wmo_ddhhmm(t) or ""))
    return texts


def _wmo_ddhhmm(text: str) -> str | None:
    m = _WMO_RE.search(text)
    return m.group("ddhhmm") if m else None


# --------------------------------------------------------------------------- parsing


def _to_int(token: str) -> int | None:
    if token in ("M", "MM"):
        return None
    return int(token)


def _norm_time(token: str | None) -> str | None:
    """Normalise the LST time column (``355 PM`` / ``3:25 PM``) to ``H:MM AM``; ``None`` if absent."""
    if not token:
        return None
    m = _TIME_RE.match(token)
    if not m:
        return None
    return f"{int(m.group('h'))}:{m.group('m')} {m.group('ampm')}"


def _parse_extremes(section: str) -> dict:
    out: dict = {"tmax_f": None, "tmin_f": None, "tmax_time": None, "tmin_time": None,
                 "tmax_record": False, "tmin_record": False}
    seen = set()
    for line in section.splitlines():
        m = _EXTREME_RE.match(line)
        if not m:
            continue
        what = m.group("what")
        if what in seen:
            continue
        seen.add(what)
        key = "tmax" if what == "MAXIMUM" else "tmin"
        out[f"{key}_f"] = _to_int(m.group("val"))
        out[f"{key}_time"] = _norm_time(line[m.end():])
        out[f"{key}_record"] = "R" in (m.group("flag") or "")
        if len(seen) == 2:
            break
    return out


def parse_issuance_time(text: str) -> datetime | None:
    """UTC issuance time reconstructed from the ``242 AM EDT SUN AUG 30 2026`` header line."""
    m = _ISSUE_RE.search(text)
    if not m:
        return None
    hm = m.group("hm").zfill(4)
    hour12, minute = int(hm[:2]), int(hm[2:])
    hour = hour12 % 12 + (12 if m.group("ampm") == "PM" else 0)
    mon = MONTHS.get(m.group("mon").upper())
    off = TZ_ABBR.get(m.group("tz").upper())
    if mon is None or off is None:
        return None
    local = datetime(int(m.group("year")), mon, int(m.group("day")), hour, minute,
                     tzinfo=timezone(timedelta(hours=off)))
    return local.astimezone(UTC)


def parse_cli(text: str) -> dict | None:
    """Parse one CLI product. Returns ``None`` if it is not a parseable climate report.

    Keys: ``climo_date``, ``block`` (``"YESTERDAY"``/``"TODAY"``), ``tmax_f``, ``tmin_f``,
    ``tmax_time``, ``tmin_time``, ``is_final`` (block is ``YESTERDAY``), ``is_corrected``,
    ``issuance_time`` (UTC, may be ``None``), ``station_hint`` plus ``pil``, ``office``,
    ``wmo_suffix`` and the record flags.
    """
    if not text or "CLIMATE SUMMARY FOR" not in text.upper():
        return None
    title = _TITLE_RE.search(text)
    if not title:
        return None
    mon = MONTHS.get(title.group("mon").upper())
    if mon is None:
        return None

    issuance = parse_issuance_time(text)
    year = title.group("year")
    if year:
        climo_year = int(year)
    elif issuance is not None:
        # No year in the title: pick the year that puts the date within a few days of issuance.
        climo_year = issuance.year
        cand = date(climo_year, mon, int(title.group("day")))
        if (cand - issuance.date()).days > 180:
            climo_year -= 1
        elif (issuance.date() - cand).days > 180:
            climo_year += 1
    else:
        return None
    climo_date = date(climo_year, mon, int(title.group("day")))

    # Temperature section: the block header is the first bare YESTERDAY/TODAY line after
    # "TEMPERATURE (F)"; the section ends at the next major heading.
    tstart = text.upper().find("TEMPERATURE (F)")
    tstart = max(tstart, 0)
    bm = _BLOCK_RE.search(text, tstart)
    if not bm:
        return None
    block = bm.group(1).upper()
    endm = _SECTION_END_RE.search(text, bm.end())
    section = text[bm.end(): endm.start() if endm else bm.end() + 600]

    wmo = _WMO_RE.search(text)
    suffix = (wmo.group("suffix").strip() if wmo else "")
    corrected = suffix.startswith("CC") or "CORRECTED" in text[:600].upper()

    pil_m = re.search(r"^(CLI[A-Z0-9]{2,4})\s*$", text, re.MULTILINE)

    out = {
        "climo_date": climo_date,
        "block": block,
        "is_final": block == "YESTERDAY",
        "is_corrected": corrected,
        "issuance_time": issuance,
        "station_hint": " ".join(title.group("name").split()),
        "pil": pil_m.group(1) if pil_m else None,
        "office": wmo.group("cccc") if wmo else None,
        "wmo_suffix": suffix,
    }
    out.update(_parse_extremes(section))
    return out


# --------------------------------------------------------------------------- day lookup


def fetch_cli_day(station: Station, climo_date: date, limit: int = 50) -> dict | None:
    """First-final CLI for one station-day, or ``None`` if no ``YESTERDAY`` report exists yet.

    The returned dict is a :func:`parse_cli` result with ``product_id`` and the authoritative
    ``issuance_time`` from the API, plus ``later_versions``: every subsequent ``YESTERDAY`` report
    for the same day (used to populate the ``revised_*`` columns).
    """
    _, day_end = day_bounds_utc(station, climo_date)
    products = [p for p in list_cli_products(station, limit=limit) if p["issuance_time"] >= day_end]
    products.sort(key=lambda p: p["issuance_time"])

    finals: list[dict] = []
    for p in products:
        try:
            text = get_product_text(p["product_id"])
        except Exception:  # pragma: no cover - network flake on a single product
            continue
        parsed = parse_cli(text)
        if not parsed or parsed["climo_date"] != climo_date or parsed["block"] != "YESTERDAY":
            continue
        parsed["product_id"] = p["product_id"]
        parsed["issuance_time"] = p["issuance_time"]
        parsed["office"] = parsed["office"] or p["office"]
        finals.append(parsed)

    if not finals:
        return None
    first = finals[0]
    first["later_versions"] = finals[1:]
    return first


def cli_history_by_day(station: Station, start: date, end: date) -> dict[date, dict]:
    """First-final CLI per climatological day from the IEM archive, for ``[start, end]``.

    The archive is queried one day past ``end`` so that the report issued the following morning is
    included. Products are keyed by their own parsed ``climo_date``.
    """
    texts = iem_cli_history(station.cli_pil, start, end + timedelta(days=1))
    parsed = []
    for t in texts:
        p = parse_cli(t)
        if p and p["block"] == "YESTERDAY" and start <= p["climo_date"] <= end:
            p["product_id"] = _iem_product_id(t)
            parsed.append(p)
    parsed.sort(key=lambda p: (p["climo_date"], p["issuance_time"] or datetime.min.replace(tzinfo=UTC)))

    out: dict[date, dict] = {}
    for p in parsed:
        cur = out.get(p["climo_date"])
        if cur is None:
            p["later_versions"] = []
            out[p["climo_date"]] = p
        else:
            cur["later_versions"].append(p)
    return out


def _iem_product_id(text: str) -> str:
    """IEM-style product key ``YYYYMMDDHHMM-CCCC-TTAAII-PIL`` (best effort)."""
    wmo = _WMO_RE.search(text)
    issuance = parse_issuance_time(text)
    pil_m = re.search(r"^(CLI[A-Z0-9]{2,4})\s*$", text, re.MULTILINE)
    if not (wmo and issuance and pil_m):
        return ""
    return f"{issuance:%Y%m%d%H%M}-{wmo.group('cccc')}-{wmo.group('ttaaii')}-{pil_m.group(1)}"
