"""Tests for the JSON export, the status report and the static site generator (DESIGN §6)."""

from __future__ import annotations

import json
import re
from xml.etree import ElementTree

import pandas as pd
import pytest

from castcheck import METHODOLOGY_VERSION
from castcheck import status as status_mod
from castcheck.api import LEADERBOARD_VIEWS, export_api, openapi_document
from castcheck.config import ModelSpec, Station
from castcheck.site.build import (
    FAIRNESS,
    FAIRNESS_BANNER,
    SITE_URL,
    SUBVIEWS,
    VIEWS,
    build_site,
    citation,
    citation_long,
    next_update,
    view_slug,
)
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
    assert latest["methodology_version"] == METHODOLOGY_VERSION
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


def test_every_response_carries_the_documented_envelope(built):
    out, _, _, _ = built
    api = out / "api" / "v1"
    for rel in ("scores/latest.json", "scores/leaderboard.json", "pairwise/latest.json",
                "stations.json", "models.json",
                "leaderboard/90d-00z-bilinear-tmax.json"):
        payload = json.loads((api / rel).read_text())
        for key in ("schema_version", "methodology_version", "generated_at", "data_through",
                    "next_update", "window", "units", "method", "truth", "license"):
            assert key in payload, f"{rel} is missing {key}"
        assert set(payload["window"]) == {"type", "days", "start", "end"}
        assert payload["units"]["mae"] == "degC"
        assert set(payload["method"]) >= {"ci", "resamples", "level", "block", "ref"}
        assert "source" in payload["truth"]


def test_every_score_row_carries_a_permalink(built):
    out, _, _, _ = built
    latest = json.loads((out / "api" / "v1" / "scores" / "latest.json").read_text())
    i = latest["columns"].index("permalink")
    assert all(r[i].startswith("/station/") and r[i].endswith("/") for r in latest["rows"][:50])
    board = json.loads(
        (out / "api" / "v1" / "leaderboard" / "90d-00z-bilinear-tmax.json").read_text())
    assert board["view"] == {"window": "90d", "init_hour": 0, "method": "bilinear",
                            "variable": "tmax",
                            "page": f"{SITE_URL}/v/90d-00z-bilinear-tmax/"}
    assert board["results"]
    for r in board["results"]:
        assert r["permalink"].startswith("/station/ALL/model/")
        assert (r["rank"] is None) == (not r["ranked"])
        assert not r["ranked"] or r["n"] >= MIN_N


def test_one_leaderboard_file_per_site_view(built):
    out, _, _, _ = built
    files = sorted(p.name for p in (out / "api" / "v1" / "leaderboard").glob("*.json"))
    assert len(files) == len(LEADERBOARD_VIEWS) == 32
    assert "all-12z-nearest-tmin.json" in files


def test_openapi_document_describes_the_endpoints(built):
    out, _, _, _ = built
    doc = json.loads((out / "api" / "v1" / "openapi.json").read_text())
    assert doc["openapi"].startswith("3.1")
    assert doc == openapi_document()
    for path in ("/scores/latest.json", "/leaderboard/{view}.json",
                 "/scores/{station}/{model}/{lead}.json", "/status.json"):
        assert path in doc["paths"], path
    assert doc["servers"][0]["url"].endswith("/api/v1")


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
    assert report["uptime"]["window_days"] == status_mod.DEFAULT_DAYS
    assert report["last_run"] == report["generated_at"]
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
        "v/90d-12z-bilinear-tmax/index.html",
        "v/30d-00z-nearest-tmin/index.html",
        "v/all-12z-nearest-tmin/index.html",
        "methodology/index.html",
        "status/index.html",
        "data/index.html",
        "api/v1/index.html",
        "stations/index.html",
        "models/index.html",
        "feed.xml",
        "_headers",
        "station/KAAA/index.html",
        "station/KAAA/v/365d-12z/index.html",
        "station/ALL/index.html",
        "model/warm1/index.html",
        "model/persistence/index.html",
        "station/KAAA/model/warm1/lead/1/index.html",
        "station/KAAA/model/warm1/lead/1/errors.csv",
        "station/ALL/model/warm1/lead/3/index.html",
        "data/scores_latest.csv",
        "data/daily_errors.csv.gz",
        "data/stations.csv",
        "data/models.csv",
        "assets/site.css",
        "assets/chart.js",
    ):
        assert (out / rel).exists(), rel
    assert counts["leaderboards"] == len(VIEWS) == 32
    assert len(SUBVIEWS) == 8
    assert counts["permalinks"] > 0
    assert counts["pages"] == counts["leaderboards"] + counts["permalinks"] \
        + len(SUBVIEWS) * counts["stations"] + len(SUBVIEWS) * counts["models"] + 6


def test_every_leaderboard_view_is_a_static_page_with_working_switchers(built):
    out, _, _, _ = built
    for window, init_hour, method, variable in VIEWS:
        slug = view_slug(window, init_hour, method, variable)
        rel = "index.html" if slug == "90d-00z-bilinear-tmax" else f"v/{slug}/index.html"
        html = (out / rel).read_text(encoding="utf-8")
        assert 'aria-current="page"' in html, rel
        # the switcher offers plain links, never a script-driven control
        assert 'href="/v/' in html or slug == "90d-00z-bilinear-tmax"
        assert "<select" not in html and "onclick" not in html


def test_every_page_carries_the_fairness_banner(built):
    out, _, _, _ = built
    pages = list(out.rglob("index.html"))
    assert len(pages) > 20
    for p in pages:
        html = p.read_text(encoding="utf-8")
        assert FAIRNESS_BANNER in html, p
        assert "/methodology/#7-fairness-statement" in html, p
    # the banner is the short form; the full statement stays available for quoting
    assert "without MOS, bias correction" in FAIRNESS


def test_assets_are_content_hashed_so_a_deploy_is_not_served_stale(built):
    out, _, _, _ = built
    html = (out / "index.html").read_text(encoding="utf-8")
    m = re.search(r'href="(/assets/site\.[0-9a-f]{10}\.css)"', html)
    assert m, "the stylesheet link must carry a content hash"
    assert (out / m.group(1).lstrip("/")).exists()
    assert re.search(r'src="/assets/chart\.[0-9a-f]{10}\.js"', html)
    # the plain names stay reachable for anyone who linked them
    assert (out / "assets" / "site.css").exists()
    assert (out / "assets" / "chart.js").exists()


def test_station_and_model_indexes_list_the_registry(built):
    out, _, _, _ = built
    stations = (out / "stations" / "index.html").read_text(encoding="utf-8")
    for st in STATION_OBJS:
        assert f'href="/station/{st.id}/"' in stations
        assert st.name in stations
        assert st.cli_pil in stations
    assert "America/New_York" in stations and "UTC-5" in stations
    models = (out / "models" / "index.html").read_text(encoding="utf-8")
    for m in MODEL_OBJS:
        assert f'href="/model/{m.model_id}/"' in models
    assert "Persistence (baseline)" in models
    assert 'href="/stations/"' in stations and 'href="/models/"' in models  # navigation


def test_model_ids_are_shown_as_human_names(built):
    out, _, _, _ = built
    html = (out / "index.html").read_text(encoding="utf-8")
    # display label as the link text, the id kept as a subtitle and a title attribute
    assert "warm1 family" in html
    assert '<span class="sub mono">warm1' in html
    assert 'title="warm1"' in html
    status = (out / "status" / "index.html").read_text(encoding="utf-8")
    assert "warm1 family" in status
    card = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    assert "<h1>Alpha Regional · warm1 family · lead day 1</h1>" in card


def test_pages_are_theme_aware_and_keyboard_reachable(built):
    out, _, _, _ = built
    html = (out / "index.html").read_text(encoding="utf-8")
    assert 'class="skip" href="#content"' in html
    assert 'id="theme-toggle"' in html
    assert "castcheck-theme" in html  # the pre-paint script that honours a stored choice
    css = (out / "assets" / "site.css").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in css
    assert ':root[data-theme="dark"]' in css
    assert ":focus-visible" in css


def test_figures_carry_a_title_and_an_equivalent_table(built):
    out, _, _, _ = built
    html = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    assert html.count("<svg") >= 2
    # every SVG names itself for assistive technology
    for chunk in html.split("<svg")[1:]:
        assert 'role="img"' in chunk[:400]
        assert 'aria-label="' in chunk[:400]
        assert "<title>" in chunk[:600]
    # and the same numbers exist as text
    assert "The same series as a table" in html
    assert "Bin counts as a table" in html
    assert "<details" in html


def test_permalink_page_is_the_whole_score_card(built):
    out, _, _, _ = built
    html = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    short = citation("KAAA", "warm1", 1, "2026-08-29")
    assert short in html
    assert f"{SITE_URL}/station/KAAA/model/warm1/lead/1/" in short
    assert f"methodology v{METHODOLOGY_VERSION}" in short
    assert citation_long("KAAA", "Alpha Regional", "warm1", 1, "2026-08-29", "2026-08-29")[:60] in html
    assert "/api/v1/scores/KAAA/warm1/1.json" in html
    assert 'href="/station/KAAA/model/warm1/lead/1/errors.csv"' in html
    assert ">Skill</abbr>" in html and "Skill, debiased" in html
    assert 'blockquote class="citation"' in html
    assert 'rel="canonical"' in html
    # all four windows are on the one page
    for window in ("30d", "90d", "all"):
        assert f"<td>{window}</td>" in html
    # the interpolation and initialization the fixture carries are named per row
    assert "<td>bilinear</td>" in html
    assert re.search(r"<td>\d\dZ</td>", html)
    # compact interval notation, e.g. "[1.9, 2.3]"
    assert re.search(r"\[-?\d+\.\d\d, -?\d+\.\d\d\]", html)


def test_permalink_csv_has_a_header_and_the_daily_errors(built):
    out, _, _, _ = built
    csv = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "errors.csv").read_text()
    lines = csv.strip().splitlines()
    assert lines[0].split(",")[-1] == "error_f"
    assert len(lines) > 1
    assert lines[1].startswith("KAAA,warm1,1,")


def test_leaderboard_ranks_by_mae_and_marks_low_n(dataset, tmp_path):
    daily, truth, scores, pairwise = dataset
    build_site(as_of="2026-08-29", out=tmp_path, scores=scores, pairwise=pairwise,
               daily=daily, truth=truth, stations=STATION_OBJS, models=MODEL_OBJS,
               permalinks=False)
    html = (tmp_path / "index.html").read_text()
    assert "How far off was each weather model?" in html
    assert html.index("exact family") < html.index("cold3 family")
    assert "Data availability" in html
    assert f"n&nbsp;&lt;&nbsp;{MIN_N}" in html
    # MAE and bias are published side by side, each with its own interval
    assert "MAE °F" in html and "Bias °F" in html
    assert "Skill, debiased" in html
    # the leader is marked and the significance legend is present
    assert "★" in html


def test_headline_states_the_conclusion_with_n_window_and_update_times(dataset, tmp_path):
    daily, truth, scores, pairwise = dataset
    build_site(as_of="2026-08-29", out=tmp_path, scores=scores, pairwise=pairwise,
               daily=daily, truth=truth, stations=STATION_OBJS, models=MODEL_OBJS,
               permalinks=False)
    html = (tmp_path / "index.html").read_text()
    assert "data through" in html
    assert "next update" in html
    assert "scored days" in html or "n &lt;" in html
    assert "window" in html and "interpolation" in html


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


def test_data_page_lists_schema_units_licences_downloads_and_limitations(built):
    out, _, _, _ = built
    html = (out / "data" / "index.html").read_text()
    assert "CC BY 4.0" in html and "public domain" in html
    assert "10.1175/BAMS-D-24-0057.1" in html  # the AIWP citation
    assert "daily_forecasts" in html and "pairwise" in html
    assert "Raw model output" in html
    assert "scores_latest.csv" in html and "daily_errors.csv.gz" in html
    assert "stations.csv" in html and "models.csv" in html
    assert "degC" in html or "°C" in html
    assert "Known limitations" in html and "Changelog" in html
    assert "<th scope=\"col\">Unit</th>" in html


def test_status_page_is_an_uptime_view(built):
    out, _, _, _ = built
    html = (out / "status" / "index.html").read_text()
    assert "Model runs" in html and "Truth (first-final NWS CLI)" in html
    assert "cellbox" in html
    assert "uptime" in html  # the 90-day bars
    assert "last run" in html
    assert f"last {status_mod.DEFAULT_DAYS} days" in html


def test_headers_file_sets_cors_and_caching_for_the_api(built):
    out, _, _, _ = built
    headers = (out / "_headers").read_text()
    assert "/api/*" in headers
    assert "Access-Control-Allow-Origin: *" in headers
    assert "Cache-Control: public, max-age=3600, stale-while-revalidate=86400" in headers


def test_feed_is_valid_atom(built):
    out, _, _, _ = built
    xml = (out / "feed.xml").read_text()
    root = ElementTree.fromstring(xml)
    assert root.tag.endswith("feed")
    entries = [e for e in root if e.tag.endswith("entry")]
    assert entries, "expected at least one update entry"
    assert any(c.tag.endswith("summary") and c.text for c in entries[0])


def test_next_update_is_the_next_publish_slot():
    assert next_update("2026-08-30T09:00:00+00:00") == "2026-08-30T11:00:00+00:00"
    assert next_update("2026-08-30T11:00:00+00:00") == "2026-08-31T11:00:00+00:00"
    assert next_update("2026-08-30T23:30:00+00:00") == "2026-08-31T11:00:00+00:00"


def test_build_site_without_data_produces_a_page_not_a_crash(tmp_path):
    counts = build_site(as_of="2026-08-30", out=tmp_path, scores=pd.DataFrame(),
                        pairwise=pd.DataFrame(), daily=pd.DataFrame(columns=DAILY_COLUMNS),
                        truth=pd.DataFrame(columns=TRUTH_COLUMNS), stations=STATION_OBJS,
                        models=MODEL_OBJS)
    assert counts["pages"] == 7
    html = (tmp_path / "index.html").read_text()
    assert "no data" in html.lower() or "nothing to score" in html.lower()
    assert FAIRNESS_BANNER in html
    for rel in ("methodology/index.html", "status/index.html", "data/index.html",
                "api/v1/index.html",
        "stations/index.html",
        "models/index.html", "feed.xml", "_headers", "data/scores_latest.csv"):
        assert (tmp_path / rel).exists(), rel


def test_export_api_without_data_produces_empty_endpoints(tmp_path):
    written = export_api(pd.DataFrame(), pd.DataFrame(), STATION_OBJS, MODEL_OBJS,
                         out=tmp_path / "api" / "v1")
    assert written["scores/latest.json"] == 0
    payload = json.loads((tmp_path / "api" / "v1" / "scores" / "latest.json").read_text())
    assert payload["rows"] == []
    assert payload["window"]["type"] == "multiple"
    assert (tmp_path / "api" / "v1" / "leaderboard"
            / "90d-00z-bilinear-tmax.json").exists()
