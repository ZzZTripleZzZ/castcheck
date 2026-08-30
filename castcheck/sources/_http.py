"""Small HTTP helper shared by the byte-range GRIB adapters (`ecmwf.py`, `gfs.py`).

Every call returns an :class:`HttpResult` instead of raising, so that adapters can turn a failure into
an explicit ``missing_reason`` row (DESIGN §0 "explicit missing"). Retries use exponential backoff and
apply only to transient conditions (timeouts, connection errors, 5xx, 429); a 404 is final.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass

import requests

from ..config import USER_AGENT

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
# The public ECMWF S3 mirror answers ~50% of requests with 503 SlowDown during busy periods (measured
# 2026-08-30 even for single, spaced requests), so we retry patiently and cap the backoff at 30 s.
DEFAULT_RETRIES = int(os.environ.get("CASTCHECK_HTTP_RETRIES", "10"))
BACKOFF_CAP_S = float(os.environ.get("CASTCHECK_HTTP_BACKOFF_CAP", "30"))
MAX_WORKERS = 8

_local = threading.local()


def session() -> requests.Session:
    """A per-thread `requests.Session` carrying the project User-Agent."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _local.session = s
    return s


@dataclass(frozen=True)
class HttpResult:
    url: str
    status: int | None
    content: bytes | None
    reason: str  # "" when ok, else a missing_reason token

    @property
    def ok(self) -> bool:
        return self.reason == "" and self.content is not None

    @property
    def text(self) -> str:
        return (self.content or b"").decode("utf-8", errors="replace")


def _reason_for_status(status: int) -> str:
    if status == 404:
        return "http_404"
    if status == 403:
        return "http_403"
    return f"http_{status}"


def fetch(
    url: str,
    *,
    byte_range: tuple[int, int] | tuple[int, None] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    head: bool = False,
) -> HttpResult:
    """GET (or HEAD) `url`, optionally a byte range, with bounded retries.

    `byte_range` is an inclusive ``(start, end)`` pair as used by the HTTP ``Range`` header; ``end``
    may be ``None`` for an open-ended range (last message of a GRIB file).
    """
    headers: dict[str, str] = {}
    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    last = HttpResult(url, None, None, "no_request")
    for attempt in range(retries):
        try:
            r = session().request("HEAD" if head else "GET", url, headers=headers, timeout=timeout)
        except requests.Timeout:
            last = HttpResult(url, None, None, "timeout")
        except requests.RequestException as exc:  # DNS, connection reset, TLS ...
            log.debug("network error on %s: %s", url, exc)
            last = HttpResult(url, None, None, "network_error")
        else:
            if r.status_code in (200, 206):
                return HttpResult(url, r.status_code, b"" if head else r.content, "")
            last = HttpResult(url, r.status_code, None, _reason_for_status(r.status_code))
            if r.status_code not in (429, 500, 502, 503, 504):
                return last
        if attempt < retries - 1:
            # exponential backoff with jitter; S3 answers a burst of range GETs with 503 SlowDown
            time.sleep(min(2**attempt, BACKOFF_CAP_S) * (1.0 + random.random() * 0.3))
    log.warning("giving up on %s after %d attempts: %s", url, retries, last.reason)
    return last
