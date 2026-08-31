"""Unit tests for the NCEP GFS adapter (offline) plus one live network test."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from castcheck.config import Station, load_stations, model_by_id
from castcheck.sources import gfs
from castcheck.sources.base import FetchRequest

INIT = datetime(2026, 8, 30, 0, tzinfo=UTC)

# Real lines from gfs.20260830/00/atmos/gfs.t00z.pgrb2.0p25.f006.idx (2 mb line kept on purpose: the
# level string must be matched exactly, not by prefix).
IDX_TEXT = """\
106:80991305:d=2026083000:TMP:2 mb:6 hour fcst:
581:419702595:d=2026083000:TMP:2 m above ground:6 hour fcst:
585:422756358:d=2026083000:APTMP:2 m above ground:6 hour fcst:
586:423312699:d=2026083000:TMAX:2 m above ground:0-6 hour max fcst:
587:423804847:d=2026083000:TMIN:2 m above ground:0-6 hour min fcst:
"""


def test_parse_idx_byte_ranges():
    recs = gfs.parse_idx(IDX_TEXT)
    assert len(recs) == 5
    assert recs[0]["offset"] == 80991305
    # a record ends one byte before the next record starts
    assert recs[1]["end"] == 422756358 - 1
    assert recs[3]["offset"] == 423312699 and recs[3]["end"] == 423804847 - 1
    # the last record is open-ended
    assert recs[-1]["end"] is None


def test_find_record_matches_level_exactly():
    recs = gfs.parse_idx(IDX_TEXT)
    r = gfs.find_record(recs, "TMP", gfs.LEVEL_2M, "6 hour fcst")
    assert r is not None and r["offset"] == 419702595  # not the "2 mb" record
    assert gfs.find_record(recs, "TMAX", gfs.LEVEL_2M, "0-6 hour max fcst")["n"] == 586
    assert gfs.find_record(recs, "TMAX", gfs.LEVEL_2M, "6-12 hour max fcst") is None


def test_object_url():
    assert gfs.object_url(gfs.AWS_BASE, INIT, 6, ".idx") == (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260830/00/atmos/"
        "gfs.t00z.pgrb2.0p25.f006.idx"
    )
    assert gfs.object_url(gfs.AWS_BASE, INIT, 240) == (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260830/00/atmos/"
        "gfs.t00z.pgrb2.0p25.f240"
    )


def test_plan_tasks():
    tasks = gfs.plan_tasks(model_by_id("gfs"))
    def steps(v):
        return sorted({t.step for t in tasks if t.variable == v})

    assert steps("t2") == list(range(6, 241, 6))
    assert steps("tmax6") == list(range(6, 241, 6))
    assert steps("tmin6") == list(range(6, 241, 6))
    assert {t.bucket_h for t in tasks if t.variable == "t2"} == {0}
    assert {t.bucket_h for t in tasks if t.variable != "t2"} == {6}
    by_step = {t.variable: t for t in tasks if t.step == 12}
    assert by_step["tmax6"].desc == "6-12 hour max fcst"
    assert by_step["tmin6"].desc == "6-12 hour min fcst"
    assert by_step["t2"].desc == "12 hour fcst"


def test_missing_run_is_all_missing_not_an_exception(monkeypatch):
    model = dataclasses.replace(model_by_id("gfs"), max_h=12)
    stations = [Station("KNYC", "n", "CLINYC", "America/New_York", -5, 40.78, -73.97, 10.0)]

    def fake_fetch(url, **kwargs):
        from castcheck.sources._http import HttpResult

        return HttpResult(url, 404, None, "http_404")

    monkeypatch.setattr(gfs._http, "fetch", fake_fetch)
    res = gfs.GfsSource().fetch_run(FetchRequest(model, INIT, stations))
    assert len(res.rows) == len(gfs.plan_tasks(model)) * 2
    assert res.n_present == 0
    assert set(res.rows["missing_reason"]) == {"http_404"}


@pytest.mark.network
def test_network_one_step_values_are_sane():
    model = dataclasses.replace(model_by_id("gfs"), max_h=6)
    res = gfs.GfsSource().fetch_run(FetchRequest(model, INIT, load_stations()))
    assert res.n_present == len(res.rows) > 0
    assert set(res.rows["variable"]) == {"t2", "tmax6", "tmin6"}
    assert res.rows["value_c"].between(-60, 60).all()
    assert (res.rows["model_version"] == gfs.MODEL_VERSION).all()
