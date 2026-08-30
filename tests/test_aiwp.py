"""Unit tests for the AIWP adapter (offline) plus one live network test."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import numpy as np
import pytest

from castcheck.config import Station, load_stations, model_by_id
from castcheck.sources import aiwp
from castcheck.sources._http import HttpResult
from castcheck.sources.base import FetchRequest

INIT = datetime(2026, 8, 30, 0, tzinfo=UTC)
GRAP_IFS = model_by_id("graphcast_ifs")
KNYC = Station("KNYC", "n", "CLINYC", "America/New_York", -5, 40.78333, -73.96667, 46.9)

# Real bucket-root listing, trimmed (2026-08-30). Note FOUR ships v100 and v200 side by side and
# FOUR_v100_IFS has never existed.
ROOT_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>noaa-oar-mlwp-data</Name><IsTruncated>false</IsTruncated>
  <CommonPrefixes><Prefix>AURO_v100_GFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>AURO_v100_IFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>Derived/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>FOUR_v100_GFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>FOUR_v200_GFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>FOUR_v200_IFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>GRAP_v100_GFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>GRAP_v100_IFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>PANG_v100_GFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>PANG_v100_IFS/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>colab_resources/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>parquet/</Prefix></CommonPrefixes>
</ListBucketResult>
"""

# A year listing carrying the 2023-only 06/18Z runs, which must be ignored.
YEAR_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>GRAP_v100_IFS/2023/0704/GRAP_v100_IFS_2023070400_f000_f240_06.nc</Key></Contents>
  <Contents><Key>GRAP_v100_IFS/2023/0704/GRAP_v100_IFS_2023070406_f000_f240_06.nc</Key></Contents>
  <Contents><Key>GRAP_v100_IFS/2023/0704/GRAP_v100_IFS_2023070412_f000_f240_06.nc</Key></Contents>
  <Contents><Key>GRAP_v100_IFS/2023/0704/GRAP_v100_IFS_2023070418_f000_f240_06.nc</Key></Contents>
  <Contents><Key>GRAP_v100_IFS/2023/0705/GRAP_v100_IFS_2023070500_f000_f240_06.nc</Key></Contents>
  <Contents><Key>GRAP_v100_IFS/2023/0705/somethingelse.txt</Key></Contents>
</ListBucketResult>
"""


def _xml_responder(mapping: dict[str, bytes], default_status: int = 404):
    """Build a fake `_http.fetch` that answers listing URLs from `mapping` (substring match)."""

    def fake(url, *, head=False, byte_range=None, timeout=None, retries=None):
        for needle, body in mapping.items():
            if needle in url:
                return HttpResult(url, 200, b"" if head else body, "")
        return HttpResult(url, default_status, None, f"http_{default_status}")

    return fake


# --------------------------------------------------------------------------- naming


def test_object_key_and_url():
    assert aiwp.object_key("GRAP", "v100", "IFS", INIT) == (
        "GRAP_v100_IFS/2026/0830/GRAP_v100_IFS_2026083000_f000_f240_06.nc"
    )
    assert aiwp.object_url("FOUR", "v200", "GFS", datetime(2022, 1, 1, 12, tzinfo=UTC)) == (
        "https://noaa-oar-mlwp-data.s3.amazonaws.com/"
        "FOUR_v200_GFS/2022/0101/FOUR_v200_GFS_2022010112_f000_f240_06.nc"
    )
    # a naive datetime is treated as UTC, not as local time
    naive = datetime(2026, 8, 30, 12)  # noqa: DTZ001 — the adapter must treat this as UTC
    assert aiwp.object_key("PANG", "v100", "GFS", naive).endswith("2026083012_f000_f240_06.nc")


def test_parse_key_round_trip():
    key = aiwp.object_key("AURO", "v100", "IFS", INIT)
    assert aiwp.parse_key(key) == ("AURO", "v100", "IFS", INIT)
    assert aiwp.parse_key("GRAP_v100_IFS/2023/0705/somethingelse.txt") is None
    assert aiwp.parse_key("parquet/AURO_v100_GFS_combined_all.parq/t2/refs.0.parq") is None


def test_model_version_string():
    assert aiwp.model_version("GRAP", "v100") == "GRAP_v100"


def test_expected_valid_times_includes_f000():
    vts = aiwp.AiwpSource().expected_valid_times(GRAP_IFS, INIT)
    assert len(vts) == 41
    assert vts[0] == INIT and vts[-1] == datetime(2026, 9, 9, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- listing


def test_discover_versions(monkeypatch):
    monkeypatch.setattr(aiwp._http, "fetch", _xml_responder({"delimiter=/&": ROOT_XML, "?list-type=2&prefix=&": ROOT_XML}))
    src = aiwp.AiwpSource()
    assert src.discover_versions("FOUR", "GFS") == ["v100", "v200"]
    assert src.discover_versions("FOUR", "IFS") == ["v200"]  # FOUR_v100_IFS never existed
    assert src.discover_versions("GRAP", "IFS") == ["v100"]
    assert src.discover_versions("AURO", "GFS") == ["v100"]
    assert src.discover_versions("NOPE", "GFS") == []


def test_discover_versions_survives_a_dead_bucket(monkeypatch):
    monkeypatch.setattr(aiwp._http, "fetch", _xml_responder({}, default_status=503))
    assert aiwp.AiwpSource().discover_versions("GRAP", "IFS") == []


def test_available_inits_keeps_only_00_and_12z(monkeypatch):
    monkeypatch.setattr(
        aiwp._http, "fetch", _xml_responder({"prefix=GRAP_v100_IFS/2023/": YEAR_XML, "delimiter": ROOT_XML})
    )
    got = aiwp.AiwpSource().available_inits(GRAP_IFS, date(2023, 7, 4), date(2023, 7, 4))
    assert got == [datetime(2023, 7, 4, 0, tzinfo=UTC), datetime(2023, 7, 4, 12, tzinfo=UTC)]


# --------------------------------------------------------------------------- fetch


def test_absent_run_is_all_http_404_not_an_exception(monkeypatch):
    """A nonexistent initialization must yield explicit missing rows (DESIGN §0)."""
    monkeypatch.setattr(aiwp._http, "fetch", _xml_responder({"delimiter": ROOT_XML}))  # HEAD -> 404
    model = dataclasses.replace(GRAP_IFS, max_h=12)
    res = aiwp.AiwpSource().fetch_run(FetchRequest(model, datetime(1999, 1, 1, tzinfo=UTC), [KNYC]))
    assert res.n_present == 0
    assert len(res.rows) == 3 * 2  # f000/f006/f012 x bilinear/nearest
    assert set(res.rows["missing_reason"]) == {"http_404"}
    assert set(res.rows["variable"]) == {"t2"}
    assert res.rows["value_c"].isna().all()


def test_wrong_source_is_a_programming_error():
    with pytest.raises(ValueError):
        aiwp.AiwpSource().fetch_run(FetchRequest(model_by_id("gfs"), INIT, [KNYC]))


# ---- synthetic NetCDF, exercising the fill-value / unit / time-check logic ----


class _Var:
    def __init__(self, data, attrs=None):
        self._d = np.asarray(data)
        self.attrs = attrs or {}

    @property
    def shape(self):
        return self._d.shape

    def __getitem__(self, item):
        return self._d[item]


class _DS:
    """Minimal stand-in for an `h5netcdf.File` covering what `_read_all_layers` touches."""

    def __init__(self, variables):
        self.variables = variables


def _synthetic_ds(n_steps: int, *, fill_first: bool = True, times=None):
    lats = np.array([41.0, 40.75, 40.5], dtype="float32")  # descending, as in the real files
    lons = np.array([285.75, 286.0, 286.25], dtype="float32")  # 0..360 convention
    t2 = np.full((n_steps, 3, 3), 293.15, dtype="float32")
    if fill_first:
        t2[0, :, :] = 9.96921e36  # the real f000 fill value
    if times is None:
        times = [int(INIT.timestamp()) + 6 * 3600 * i for i in range(n_steps)]
    return _DS(
        {
            "t2": _Var(t2, {"units": b"K", "long_name": "2 metre temperature"}),
            "latitude": _Var(lats),
            "longitude": _Var(lons),
            "time": _Var(np.asarray(times, dtype="int32"), {"units": "seconds since 1970-1-1"}),
        }
    )


def _read(ds, model, valid_times, notes=None):
    src = aiwp.AiwpSource()
    frames = src._read_all_layers(
        ds, model, "GRAP_v100", INIT, valid_times, [KNYC], "http://x", notes if notes is not None else []
    )
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def test_f000_fill_value_is_missing_and_later_steps_convert_kelvin():
    model = dataclasses.replace(GRAP_IFS, max_h=12)
    vts = aiwp.AiwpSource().expected_valid_times(model, INIT)
    rows = _read(_synthetic_ds(3), model, vts)
    assert len(rows) == 3 * 2
    f000 = rows[rows.lead_h == 0]
    assert set(f000["missing_reason"]) == {"fill_value"}
    assert f000["value_c"].isna().all()
    later = rows[rows.lead_h > 0]
    assert set(later["missing_reason"]) == {""}
    assert np.allclose(later["value_c"], 20.0)  # 293.15 K
    assert set(later["method"]) == {"bilinear", "nearest"}
    assert set(rows["bucket_h"]) == {0} and set(rows["variable"]) == {"t2"}


def test_steps_beyond_the_file_are_no_field():
    model = dataclasses.replace(GRAP_IFS, max_h=18)
    vts = aiwp.AiwpSource().expected_valid_times(model, INIT)  # 4 steps
    rows = _read(_synthetic_ds(3), model, vts)  # file only has 3
    assert set(rows[rows.lead_h == 18]["missing_reason"]) == {"no_field"}


def test_missing_variable_makes_the_whole_run_missing():
    model = dataclasses.replace(GRAP_IFS, max_h=6)
    ds = _synthetic_ds(2)
    del ds.variables["t2"]
    rows = _read(ds, model, aiwp.AiwpSource().expected_valid_times(model, INIT))
    assert set(rows["missing_reason"]) == {"no_field"}


def test_time_coordinate_is_cross_checked_against_init_plus_step():
    model = dataclasses.replace(GRAP_IFS, max_h=12)
    vts = aiwp.AiwpSource().expected_valid_times(model, INIT)
    bad = [int(INIT.timestamp()) + 6 * 3600 * i for i in range(3)]
    bad[2] += 3600  # f012 layer is stamped one hour off
    notes: list[str] = []
    rows = _read(_synthetic_ds(3, times=bad), model, vts, notes)
    assert set(rows[rows.lead_h == 12]["missing_reason"]) == {"time_mismatch"}
    assert set(rows[rows.lead_h == 6]["missing_reason"]) == {""}
    assert any("time[2]" in n for n in notes)


def test_fill_detection_uses_kelvin_sanity_bounds():
    assert aiwp.grid.is_fill(9.96921e36)
    assert aiwp.grid.is_fill(float("nan"))
    assert not aiwp.grid.is_fill(293.15)


# --------------------------------------------------------------------------- network


@pytest.mark.network
def test_network_one_run_values_are_sane():
    model = dataclasses.replace(GRAP_IFS, max_h=6)  # f000 + f006 only
    stations = load_stations()
    res = aiwp.AiwpSource().fetch_run(FetchRequest(model, INIT, stations))
    assert len(res.rows) == 2 * len(stations) * 2
    assert (res.rows["model_version"] == "GRAP_v100").all()
    # f000 is the fill value in every AIWP file; f006 must be a real temperature everywhere
    assert set(res.rows[res.rows.lead_h == 0]["missing_reason"]) == {"fill_value"}
    good = res.rows[res.rows.lead_h == 6]
    assert set(good["missing_reason"]) == {""}
    assert good["value_c"].between(-60, 60).all()
    assert good["station_id"].nunique() == len(stations)
