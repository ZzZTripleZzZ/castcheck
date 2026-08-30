"""Tests for the JSON export, the status report and the static site generator (DESIGN §6)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from castcheck import status as status_mod
from castcheck.api import export_api
from castcheck.config import ModelSpec, Station
from castcheck.site.build import FAIRNESS, SITE_URL, build_site, citation
from castcheck.store import DAILY_COLUMNS, TRUTH_COLUMNS
from castcheck.verify import ALL_STATIONS, MIN_N, score

from .test_verify import make_daily, make_truth

STATION_OBJS = [
    Station(id="KAAA", name="Alpha Regional", cli_pil="CLIAAA", tz="America/New_York",
            std_offset_h=-5, lat=40.0, lon=-74.0, elev_m=10.0),
    Station(id="KBBB", name="Bravo Intl", cli_pil="CLIBBB", tz="America/Chicago",
            std_offset_h=-6, lat=41.0, lon=-88.0, elev_m=200.0),
]
MODEL_OBJS = [
    ModelSpec(model_id=mid, family=f"{mid} family", source="ecmwf", product="oper",
              init_field=None, inits=(0, 12), step_h=6, max_h=240, native_extremes=())
    for mid in ("exact", "warm1", "split", "cold3", "noisy")
]


@pytest.fixture(scope="module")
def dataset():
    truth = make_truth()
    daily = make_daily(truth)
    scores, pairwise = score(daily, truth, windows=(30, 90, None), n_boot=200, seed=0)
    return daily, truth, scores, pairwise


@pytest.fixture(scope="module")
def built(dataset, tmp_path_factory):
    daily, truth, scores, pairwise = dataset
    out = tmp_path_factory.mktemp("site")
    report = status_mod.build(as_of="2026-08-30", values=pd.DataFrame(), truth=truth,
                              stations=STATION_OBJS, models=MODEL_OBJS)
    written = export_api(scores, pairwise, STATION_OBJS, MODEL_OBJS, out=out / "api" / "v1",
                         daily=daily, truth=truth, status=report)
    counts = build_site(as_of="2026-08-29", out=out, scores=scores, pairwise=pairwise,
                        daily=daily, truth=truth, stations=STATION_OBJS, models=MODEL_OBJS,
                        status_report=report, api_written=written)
    return out, counts, written, report


# ------------------------------------------------------------------------------------ API

def test_api_endpoints_exist(built):
    out, _, written, _ = built
    api = out / "api" / "v1"
    for rel in ("stations.json", "models.json", "status.json",
                "scores/latest.json", "scores/leaderboard.json", "pairwise/latest.json"):
        assert (api / rel).exists(), rel
    latest = json.loads((api / "scores" / "latest.json").read_text())
    assert latest["methodology_version"] == "0.1"
    assert "columns" in latest and "rows" in latest
    assert len(latest["rows"]) == written["scores/latest.json"]
    assert "station_id" in latest["columns"] and "mae_ci_low" in latest["columns"]
    board = json.loads((api / "scores" / "leaderboard.json").read_text())
    st_col = board["columns"].index("station_id")
    assert {r[st_col] for r in board["rows"]} == {ALL_STATIONS}


def test_permalink_json_has_scores_pairwise_and_series(built):
    out, _, _, _ = built
    card = json.loads((out / "api" / "v1" / "scores" / "KAAA" / "warm1" / "1.json").read_text())
    assert card["station_id"] == "KAAA" and card["model_id"] == "warm1" and card["lead_day"] == 1
    assert card["permalink"] == "/station/KAAA/model/warm1/lead/1/"
    assert len(card["scores"]["rows"]) > 0
    assert len(card["pairwise"]["rows"]) > 0
    assert card["series"], "expected a daily error series"
    s = card["series"][0]
    assert len(s["dates"]) == len(s["err_c"]) <= card["series_days"]


def test_stations_json_includes_the_all_pseudo_station(built):
    out, _, _, _ = built
    payload = json.loads((out / "api" / "v1" / "stations.json").read_text())
    ids = {s["id"] for s in payload["stations"]}
    assert ids == {"KAAA", "KBBB", ALL_STATIONS}


# --------------------------------------------------------------------------------- status

def test_status_report_and_exit_code(built):
    _, _, _, report = built
    assert report["as_of"] == "2026-08-30"
    assert report["n_stations"] == 2
    assert len(report["dates"]) == status_mod.DEFAULT_DAYS
    # no forecast values were supplied, so every model run in the window is a gap
    assert report["n_gaps"] > 0
    assert report["ok"] is False
    assert status_mod.exit_code(report) == status_mod.EXIT_GAPS
    assert all(m["expected_steps"] == 40 for m in report["models"])


def test_status_exit_code_is_zero_when_nothing_is_missing_today():
    report = {"n_current_gaps": 0}
    assert status_mod.exit_code(report) == status_mod.EXIT_OK


# ----------------------------------------------------------------------------------- site

def test_key_routes_exist(built):
    out, counts, _, _ = built
    for rel in (
        "index.html",
        "v/12z-bilinear/index.html",
        "v/00z-nearest/index.html",
        "methodology/index.html",
        "status/index.html",
        "data/index.html",
        "station/KAAA/index.html",
        "station/ALL/index.html",
        "model/warm1/index.html",
        "model/persistence/index.html",
        "station/KAAA/model/warm1/lead/1/index.html",
        "station/ALL/model/warm1/lead/3/index.html",
        "assets/site.css",
        "assets/chart.js",
    ):
        assert (out / rel).exists(), rel
    assert counts["permalinks"] > 0
    assert counts["pages"] == counts["leaderboards"] + counts["permalinks"] \
        + 4 * counts["stations"] + 4 * counts["models"] + 3


def test_every_page_carries_the_fairness_statement(built):
    out, _, _, _ = built
    pages = list(out.rglob("index.html"))
    assert len(pages) > 20
    for p in pages:
        assert FAIRNESS in p.read_text(encoding="utf-8"), p


def test_permalink_page_has_citation_json_link_and_intervals(built):
    out, _, _, _ = built
    html = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    expected = citation("KAAA", "warm1", 1, "2026-08-29")
    assert expected in html
    assert f"{SITE_URL}/station/KAAA/model/warm1/lead/1/" in expected
    assert "methodology v0.1" in expected
    assert "/api/v1/scores/KAAA/warm1/1.json" in html
    assert "95 % CI" in html
    assert "Skill vs persistence" in html
    assert 'blockquote class="citation"' in html
    # the chart is an enhancement; the table is the content
    assert 'data-chart="/api/v1/scores/KAAA/warm1/1.json"' in html
    assert "<table" in html


def test_leaderboard_ranks_by_mae_and_marks_low_n(dataset, tmp_path):
    daily, truth, scores, pairwise = dataset
    build_site(as_of="2026-08-29", out=tmp_path, scores=scores, pairwise=pairwise,
               daily=daily, truth=truth, stations=STATION_OBJS, models=MODEL_OBJS,
               permalinks=False)
    html = (tmp_path / "index.html").read_text()
    assert "Model leaderboard" in html
    # the exact model must appear before the deliberately bad one
    assert html.index(">exact<") < html.index(">cold3<")
    assert "Data availability" in html
    assert f"n&nbsp;&lt;&nbsp;{MIN_N}" in html


def test_low_n_rows_are_greyed(tmp_path, dataset):
    daily, truth, _, _ = dataset
    few = pd.to_datetime(daily["climo_date"]) >= pd.Timestamp("2026-08-20")
    scores, pairwise = score(daily[few], truth, windows=(None,), n_boot=100, seed=0)
    build_site(as_of="2026-08-29", out=tmp_path, scores=scores, pairwise=pairwise,
               daily=daily[few], truth=truth, stations=STATION_OBJS, models=MODEL_OBJS)
    html = (tmp_path / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    assert 'class="low-n"' in html


def test_methodology_page_renders_the_document(built):
    out, _, _, _ = built
    html = (out / "methodology" / "index.html").read_text()
    assert "Climatological day" in html
    assert 'id="23-sampled-daily-extremes-headline"' in html  # anchored from the footer
    assert "<table" in html


def test_data_page_lists_licences_and_schemas(built):
    out, _, _, _ = built
    html = (out / "data" / "index.html").read_text()
    assert "CC-BY-4.0" in html and "public domain" in html
    assert "daily_forecasts" in html and "pairwise" in html
    assert "Raw model output" in html


def test_status_page_shows_the_grid(built):
    out, _, _, _ = built
    html = (out / "status" / "index.html").read_text()
    assert "Model runs" in html and "Truth (first-final NWS CLI)" in html
    assert "cellbox" in html


def test_build_site_without_data_produces_a_page_not_a_crash(tmp_path):
    counts = build_site(as_of="2026-08-30", out=tmp_path, scores=pd.DataFrame(),
                        pairwise=pd.DataFrame(), daily=pd.DataFrame(columns=DAILY_COLUMNS),
                        truth=pd.DataFrame(columns=TRUTH_COLUMNS), stations=STATION_OBJS,
                        models=MODEL_OBJS)
    assert counts["pages"] == 4
    html = (tmp_path / "index.html").read_text()
    assert "no data" in html.lower() or "nothing to score" in html.lower()
    assert FAIRNESS in html
    for rel in ("methodology/index.html", "status/index.html", "data/index.html"):
        assert (tmp_path / rel).exists()


def test_export_api_without_data_produces_empty_endpoints(tmp_path):
    written = export_api(pd.DataFrame(), pd.DataFrame(), STATION_OBJS, MODEL_OBJS,
                         out=tmp_path / "api" / "v1")
    assert written["scores/latest.json"] == 0
    payload = json.loads((tmp_path / "api" / "v1" / "scores" / "latest.json").read_text())
    assert payload["rows"] == []
