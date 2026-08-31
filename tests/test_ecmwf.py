"""Unit tests for the ECMWF Open Data adapter (offline) plus one live network test."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from castcheck.config import Station, load_stations, model_by_id
from castcheck.sources import ecmwf
from castcheck.sources.base import FetchRequest

INIT = datetime(2026, 8, 30, 0, tzinfo=UTC)

# Two real lines copied from 20260830/00z/ifs/0p25/oper/20260830000000-6h-oper-fc.index, plus a
# pressure-level line that must not be mistaken for the surface field and a malformed line.
INDEX_TEXT = """\
{"domain": "g", "date": "20260830", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "step": "6", "levtype": "sfc", "param": "2t", "_offset": 22287774, "_length": 649771}
{"domain": "g", "date": "20260830", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "levtype": "sfc", "step": "6", "param": "mx2t3", "_offset": 2080577, "_length": 631759}
{"domain": "g", "date": "20260830", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "step": "6", "levelist": "850", "levtype": "pl", "param": "t", "_offset": 1, "_length": 2}
not json at all
"""


def test_parse_index_and_find_entry():
    entries = ecmwf.parse_index(INDEX_TEXT)
    assert len(entries) == 3  # the malformed line is skipped, not fatal
    t2 = ecmwf.find_entry(entries, "2t")
    assert t2 is not None and t2["_offset"] == 22287774 and t2["_length"] == 649771
    assert ecmwf.find_entry(entries, "mx2t3")["_offset"] == 2080577
    assert ecmwf.find_entry(entries, "mn2t3") is None  # absent -> caller writes a no_field row
    # a pressure-level entry must never be returned as a surface field
    assert ecmwf.find_entry(entries, "t") is None


def test_object_url_ifs_and_aifs():
    ifs = model_by_id("ifs_hres")
    aifs = model_by_id("aifs_single")
    assert ecmwf.object_url(ecmwf.PORTAL_BASE, ifs, INIT, 6) == (
        "https://data.ecmwf.int/forecasts/20260830/00z/ifs/0p25/oper/"
        "20260830000000-6h-oper-fc.grib2"
    )
    assert ecmwf.object_url(ecmwf.MIRROR_BASE, ifs, INIT, 150, "index") == (
        "https://ecmwf-forecasts.s3.amazonaws.com/20260830/00z/ifs/0p25/oper/"
        "20260830000000-150h-oper-fc.index"
    )
    assert ecmwf.object_url(ecmwf.PORTAL_BASE, aifs, INIT, 240) == (
        "https://data.ecmwf.int/forecasts/20260830/00z/aifs-single/0p25/oper/"
        "20260830000000-240h-oper-fc.grib2"
    )


def test_plan_tasks_ifs_step_coverage():
    tasks = ecmwf.plan_tasks(model_by_id("ifs_hres"))
    def steps(v):
        return sorted({t.step for t in tasks if t.variable == v})

    assert steps("t2") == list(range(6, 241, 6))
    # 3-hourly mx2t3/mn2t3 out to 144 h — including the odd steps, which the 6-hourly sampling would
    # miss but derive.py needs to cover a whole climatological day
    assert steps("mx2t3") == list(range(3, 145, 3))
    assert steps("mn2t3") == list(range(3, 145, 3))
    assert 9 in steps("mx2t3") and 21 in steps("mx2t3")
    assert steps("mx2t6") == list(range(150, 241, 6))
    assert steps("mn2t6") == list(range(150, 241, 6))
    buckets = {t.variable: t.bucket_h for t in tasks}
    assert buckets == {"t2": 0, "mx2t3": 3, "mn2t3": 3, "mx2t6": 6, "mn2t6": 6}


def test_plan_tasks_aifs_has_no_native_extremes():
    tasks = ecmwf.plan_tasks(model_by_id("aifs_single"))
    assert {t.variable for t in tasks} == {"t2"}
    assert sorted(t.step for t in tasks) == list(range(6, 241, 6))


def test_plan_tasks_variable_filter():
    tasks = ecmwf.plan_tasks(model_by_id("ifs_hres"), variables=("t2",))
    assert {t.variable for t in tasks} == {"t2"}


def test_missing_run_is_all_missing_not_an_exception(monkeypatch):
    """A run that does not exist yields explicit http_404 rows for every planned value."""
    model = dataclasses.replace(model_by_id("ifs_hres"), max_h=12)
    stations = [Station("KNYC", "n", "CLINYC", "America/New_York", -5, 40.78, -73.97, 10.0)]

    def fake_fetch(url, **kwargs):
        from castcheck.sources._http import HttpResult

        return HttpResult(url, 404, None, "http_404")

    monkeypatch.setattr(ecmwf._http, "fetch", fake_fetch)
    res = ecmwf.EcmwfSource().fetch_run(FetchRequest(model, INIT, stations))
    n_tasks = len(ecmwf.plan_tasks(model))
    assert len(res.rows) == n_tasks * 2  # bilinear + nearest per station
    assert res.n_present == 0
    assert set(res.rows["missing_reason"]) == {"http_404"}
    assert res.rows["value_c"].isna().all()


@pytest.mark.network
def test_network_one_step_values_are_sane():
    """One real IFS step: values in physical range and no missing rows."""
    model = dataclasses.replace(model_by_id("ifs_hres"), max_h=6)
    res = ecmwf.EcmwfSource().fetch_run(
        FetchRequest(model, INIT, load_stations(), variables=("t2",))
    )
    assert res.n_present == len(res.rows) > 0
    assert res.rows["value_c"].between(-60, 60).all()
    assert res.rows["model_version"].str.startswith("ifs-gpid").all()
    assert res.rows["source_url"].str.endswith("20260830000000-6h-oper-fc.grib2").all()
