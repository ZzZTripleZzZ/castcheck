"""Tests for `castcheck.sources._http`: retry policy and per-host backpressure (DESIGN §7.3).

No sockets: `requests.Session.request` is replaced by a scripted stub, and `time.sleep` by a
recorder, so the pacing decisions are asserted directly instead of waited for.
"""

from __future__ import annotations

import requests

from castcheck.sources import _http


class _Resp:
    def __init__(self, status: int, content: bytes = b"ok"):
        self.status_code = status
        self.content = content


def _script(monkeypatch, responses):
    """Answer successive requests from `responses` (status ints or exceptions); record the URLs."""
    seen: list[str] = []
    queue = list(responses)

    def fake(self, method, url, headers=None, timeout=None):
        seen.append(url)
        item = queue.pop(0) if queue else 200
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    monkeypatch.setattr(requests.Session, "request", fake)
    return seen


def _no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(_http.time, "sleep", slept.append)
    return slept


def _fresh(monkeypatch, min_interval: float = 0.0):
    """Reset the shared host table and shrink the pacing floor so tests do not wait on it."""
    _http.reset_hosts()
    monkeypatch.setattr(_http, "MIN_INTERVAL_S", min_interval)


# --------------------------------------------------------------------------- retry policy


def test_200_returns_content_without_retrying(monkeypatch):
    _fresh(monkeypatch)
    seen = _script(monkeypatch, [200])
    _no_sleep(monkeypatch)
    res = _http.fetch("https://example.com/a")
    assert res.ok and res.content == b"ok" and len(seen) == 1


def test_404_is_final_and_never_retried(monkeypatch):
    _fresh(monkeypatch)
    seen = _script(monkeypatch, [404, 200])
    _no_sleep(monkeypatch)
    res = _http.fetch("https://example.com/missing", retries=5)
    assert res.reason == "http_404"
    assert len(seen) == 1  # one attempt only


def test_503_is_retried_up_to_the_limit_then_gives_up(monkeypatch):
    _fresh(monkeypatch)
    seen = _script(monkeypatch, [503] * 10)
    _no_sleep(monkeypatch)
    res = _http.fetch("https://example.com/busy", retries=3)
    assert res.reason == "http_503"
    assert len(seen) == 3
    assert _http.host_stats()["example.com"]["failures"] == 1


def test_503_then_success(monkeypatch):
    _fresh(monkeypatch)
    _script(monkeypatch, [503, 503, 200])
    _no_sleep(monkeypatch)
    assert _http.fetch("https://example.com/x", retries=5).ok


def test_timeout_and_connection_errors_are_retried(monkeypatch):
    _fresh(monkeypatch)
    _script(monkeypatch, [requests.Timeout(), requests.ConnectionError(), 200])
    _no_sleep(monkeypatch)
    assert _http.fetch("https://example.com/x", retries=5).ok


def test_a_byte_range_is_sent_as_an_inclusive_header(monkeypatch):
    _fresh(monkeypatch)
    _no_sleep(monkeypatch)
    captured: dict = {}

    def fake(self, method, url, headers=None, timeout=None):
        captured.update(headers or {})
        return _Resp(206, b"grib")

    monkeypatch.setattr(requests.Session, "request", fake)
    assert _http.fetch("https://example.com/g", byte_range=(10, 19)).content == b"grib"
    assert captured["Range"] == "bytes=10-19"
    _http.fetch("https://example.com/g", byte_range=(10, None))
    assert captured["Range"] == "bytes=10-"


# --------------------------------------------------------------------------- per-host pacing


def test_a_throttling_host_slows_down_and_the_penalty_is_shared(monkeypatch):
    """503 raises the host's minimum interval so every worker touching it backs off, not just one."""
    _fresh(monkeypatch, min_interval=0.001)
    _script(monkeypatch, [503, 503, 200])
    _no_sleep(monkeypatch)
    _http.fetch("https://slow.example.com/x", retries=5)

    stats = _http.host_stats()["slow.example.com"]
    assert stats["throttled"] == 2
    assert stats["retries"] == 2
    assert stats["interval_s"] > _http.MIN_INTERVAL_S


def test_hosts_are_paced_independently(monkeypatch):
    _fresh(monkeypatch, min_interval=0.001)
    _script(monkeypatch, [503, 200, 200])
    _no_sleep(monkeypatch)
    _http.fetch("https://slow.example.com/x", retries=3)
    _http.fetch("https://fast.example.com/y", retries=3)

    stats = _http.host_stats()
    assert stats["slow.example.com"]["throttled"] == 1
    assert stats["fast.example.com"]["throttled"] == 0
    assert stats["fast.example.com"]["interval_s"] == _http.MIN_INTERVAL_S


def test_backoff_is_bounded_by_the_cap(monkeypatch):
    _fresh(monkeypatch)
    _script(monkeypatch, [503] * 20)
    slept = _no_sleep(monkeypatch)
    _http.fetch("https://slow.example.com/x", retries=12)
    assert slept and max(slept) <= _http.BACKOFF_CAP_S * 1.3 + 1e-9


def test_summary_is_one_line_per_host(monkeypatch, caplog):
    """Failures are summarised per host at the end of a command, not warned about per attempt."""
    import logging

    _fresh(monkeypatch)
    _script(monkeypatch, [503] * 6)
    _no_sleep(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="castcheck.sources._http"):
        _http.fetch("https://slow.example.com/x", retries=3)
    assert caplog.records == []  # nothing warned while retrying

    with caplog.at_level(logging.INFO, logger="castcheck.sources._http"):
        _http.log_summary()
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1
    assert "slow.example.com" in lines[0] and "throttled" in lines[0]
