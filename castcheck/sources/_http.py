"""Small HTTP helper shared by the byte-range GRIB adapters (`ecmwf.py`, `gfs.py`).

Every call returns an :class:`HttpResult` instead of raising, so that adapters can turn a failure into
an explicit ``missing_reason`` row (DESIGN §0 "explicit missing"). Retries use exponential backoff and
apply only to transient conditions (timeouts, connection errors, 5xx, 429); a 404 is final.

Being a good upstream citizen
-----------------------------
The public ECMWF S3 mirror answers a large fraction of requests with ``503 SlowDown`` when it is
busy — measured 2026-08-30 at roughly half of even single, spaced requests. Retrying harder makes
that worse, so this module keeps *per-host* state shared by every thread in the process:

* a minimum interval between request starts (:data:`MIN_INTERVAL_S`, per host, configurable);
* a multiplicative penalty on that interval whenever the host answers 429/503, decayed back to the
  floor after a quiet period, so one throttling host slows every worker touching it instead of only
  the unlucky thread;
* a counter of retries and throttles per host, reported once by :func:`log_summary` rather than one
  ``warning`` line per failed attempt.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import requests

from ..config import USER_AGENT

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = int(os.environ.get("CASTCHECK_HTTP_RETRIES", "6"))
BACKOFF_CAP_S = float(os.environ.get("CASTCHECK_HTTP_BACKOFF_CAP", "30"))
MAX_WORKERS = 8

#: Floor on the gap between two request starts to the same host, seconds.
MIN_INTERVAL_S = float(os.environ.get("CASTCHECK_HTTP_MIN_INTERVAL", "0.05"))
#: Multiplier applied to a host's interval on every 429/503 …
THROTTLE_GROWTH = 2.0
#: … capped here, and decayed back towards :data:`MIN_INTERVAL_S` by this factor per quiet second.
THROTTLE_MAX_INTERVAL_S = float(os.environ.get("CASTCHECK_HTTP_MAX_INTERVAL", "5"))
THROTTLE_DECAY_PER_S = 0.5

RETRYABLE_STATUS = (429, 500, 502, 503, 504)

_local = threading.local()


def session() -> requests.Session:
    """A per-thread `requests.Session` carrying the project User-Agent."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _local.session = s
    return s


@dataclass
class _HostState:
    """Shared pacing state for one host. Guarded by :data:`_hosts_lock`."""

    # default_factory, not a plain default: the floor is configurable through the module constant
    # (and patched in tests), and a class-level default would freeze the value seen at import time.
    interval: float = field(default_factory=lambda: MIN_INTERVAL_S)
    #: per-host floor the interval decays back to; raised by :func:`set_min_interval`
    floor: float = field(default_factory=lambda: MIN_INTERVAL_S)
    next_start: float = 0.0
    requests: int = 0
    retries: int = 0
    throttled: int = 0
    failures: int = 0
    reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))


_hosts: dict[str, _HostState] = {}
_hosts_lock = threading.Lock()


def _host_of(url: str) -> str:
    rest = url.split("://", 1)[-1]
    return rest.split("/", 1)[0].lower()


def _state(host: str) -> _HostState:
    with _hosts_lock:
        st = _hosts.get(host)
        if st is None:
            st = _hosts[host] = _HostState()
        return st


def _acquire(host: str) -> None:
    """Block until this host's minimum inter-request interval has elapsed (shared across threads)."""
    st = _state(host)
    while True:
        with _hosts_lock:
            now = time.monotonic()
            quiet = now - st.next_start
            if quiet > 0 and st.interval > st.floor:
                # decay the penalty back towards the floor after a calm stretch
                st.interval = max(st.floor, st.interval * (THROTTLE_DECAY_PER_S ** quiet))
            wait = st.next_start - now
            if wait <= 0:
                st.next_start = now + st.interval
                st.requests += 1
                return
        time.sleep(min(wait, 1.0))


def _penalise(host: str) -> None:
    """Slow every thread that talks to `host` after it answered 429/503."""
    st = _state(host)
    with _hosts_lock:
        st.throttled += 1
        st.interval = min(max(THROTTLE_MAX_INTERVAL_S, st.floor),
                          max(st.floor, st.interval) * THROTTLE_GROWTH)
        st.next_start = max(st.next_start, time.monotonic() + st.interval)


def reset_hosts() -> None:
    """Forget all per-host pacing and counters (used by tests and between CLI commands)."""
    with _hosts_lock:
        _hosts.clear()


def set_min_interval(url_or_host: str, seconds: float) -> None:
    """Raise the floor on the gap between requests to one host (never lowers it).

    :data:`MIN_INTERVAL_S` is a global default tuned for object stores that serve byte ranges. A
    volunteer-run archive such as the Iowa Environmental Mesonet asks for a slower rate than that,
    and the polite rate is a property of the *host*, not of one call site, so it is recorded here
    where the pacing actually happens.
    """
    st = _state(_host_of(url_or_host))
    with _hosts_lock:
        st.floor = max(st.floor, float(seconds))
        st.interval = max(st.interval, st.floor)


def host_stats() -> dict[str, dict[str, int | float]]:
    """Snapshot of per-host counters: requests, retries, throttles, failures and reason histogram."""
    with _hosts_lock:
        return {
            host: {
                "requests": st.requests, "retries": st.retries, "throttled": st.throttled,
                "failures": st.failures, "interval_s": round(st.interval, 3), "reasons": dict(st.reasons),
            }
            for host, st in sorted(_hosts.items())
        }


def log_summary(level: int = logging.INFO) -> None:
    """Emit one line per host instead of one warning per failed attempt."""
    for host, s in host_stats().items():
        if s["retries"] or s["throttled"] or s["failures"]:
            log.log(level, "http %s: %d requests, %d retries, %d throttled (503/429), %d gave up %s",
                    host, s["requests"], s["retries"], s["throttled"], s["failures"], s["reasons"] or "")
        else:
            log.debug("http %s: %d requests, no retries", host, s["requests"])


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
    """GET (or HEAD) `url`, optionally a byte range, with bounded retries and per-host pacing.

    `byte_range` is an inclusive ``(start, end)`` pair as used by the HTTP ``Range`` header; ``end``
    may be ``None`` for an open-ended range (last message of a GRIB file).
    """
    headers: dict[str, str] = {}
    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    host = _host_of(url)
    st = _state(host)
    last = HttpResult(url, None, None, "no_request")
    retries = max(1, retries)
    for attempt in range(retries):
        _acquire(host)
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
            if r.status_code not in RETRYABLE_STATUS:
                with _hosts_lock:
                    st.reasons[last.reason] += 1
                return last
            _penalise(host)
        if attempt < retries - 1:
            with _hosts_lock:
                st.retries += 1
            # exponential backoff with jitter, on top of the host-wide interval penalty
            time.sleep(min(2**attempt, BACKOFF_CAP_S) * (1.0 + random.random() * 0.3))
    with _hosts_lock:
        st.failures += 1
        st.reasons[last.reason] += 1
    log.debug("giving up on %s after %d attempts: %s", url, retries, last.reason)
    return last
