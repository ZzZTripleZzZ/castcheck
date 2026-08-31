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
    HEADLINE_VARIABLE,
    REPO_URL,
    SAMPLED_VARIABLES,
    SITE_URL,
    SUBVIEWS,
    VIEWS,
    build_site,
    changelog_entries,
    citation,
    citation_long,
    next_update,
    source_commit,
    view_slug,
)
from castcheck.store import DAILY_COLUMNS, TRUTH_COLUMNS
from castcheck.verify import ALL_STATIONS, MIN_N, score

from .test_verify import make_frames, make_truth, make_truth_instant

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def text_of(html: str) -> str:
    """Visible text with tags stripped and whitespace collapsed.

    The site's markup gets redesigned; what it *says* is the contract. Assertions about wording go
    through here so a class rename or a re-wrap does not fail a test about meaning.
    """
    out = _TAGS.sub(" ", html)
    for entity, char in (("&#39;", "'"), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&nbsp;", " "), ("&minus;", "-"), ("&Delta;", "Δ"), ("&times;", "x"),
                         ("&plusmn;", "+/-"), ("&ndash;", "-")):
        out = out.replace(entity, char)
    return _WS.sub(" ", out).strip()


STATION_OBJS = [
    Station(id="KAAA", name="Alpha Regional", cli_pil="CLIAAA", tz="America/New_York",
            std_offset_h=-5, lat=40.0, lon=-74.0, elev_m=10.0),
    Station(id="KBBB", name="Bravo Intl", cli_pil="CLIBBB", tz="America/Chicago",
            std_offset_h=-6, lat=41.0, lon=-88.0, elev_m=200.0),
]
MODEL_OBJS = [
    ModelSpec(model_id=mid, family=f"{mid} family", source="ecmwf", product="oper",
              init_field=None, inits=(0, 12), step_h=6, max_h=240, native_extremes=())
    for mid in ("exact", "warm1", "cold18", "noisy")
]


def with_v03_columns(scores, pairwise):
    """The v0.3 tables as ``verify.score`` produces them (kept as a seam for the degrade test)."""
    return scores, pairwise


@pytest.fixture(scope="module")
def dataset():
    truth = make_truth()
    truth_instant = make_truth_instant()
    daily, instant = make_frames()
    scores, pairwise = score(daily, truth, instant, windows=(30, 90, None), n_boot=200, seed=0,
                             truth_instant=truth_instant)
    return daily, truth, instant, scores, pairwise


@pytest.fixture(scope="module")
def built(dataset, tmp_path_factory):
    daily, truth, instant, scores, pairwise = dataset
    out = tmp_path_factory.mktemp("site")
    report = status_mod.build(as_of="2026-08-30", values=pd.DataFrame(), truth=truth,
                              stations=STATION_OBJS, models=MODEL_OBJS, upstream=False)
    written = export_api(scores, pairwise, STATION_OBJS, MODEL_OBJS, out=out / "api" / "v1",
                         daily=daily, truth=truth, instant=instant, status=report)
    counts = build_site(as_of="2026-08-29", out=out, scores=scores, pairwise=pairwise,
                        daily=daily, truth=truth, instant=instant, stations=STATION_OBJS,
                        models=MODEL_OBJS, status_report=report, api_written=written)
    return out, counts, written, report


# ------------------------------------------------------------------------------------ API

def test_api_endpoints_exist(built):
    out, _, written, _ = built
    api = out / "api" / "v1"
    for rel in ("stations.json", "models.json", "status.json",
                "scores/index.json", "scores/latest.json", "scores/leaderboard.json",
                "pairwise/latest.json"):
        assert (api / rel).exists(), rel
    latest = json.loads((api / "scores" / "index.json").read_text())
    assert latest["methodology_version"] == METHODOLOGY_VERSION
    assert "station_id" in latest["columns"] and "mae_ci_low" in latest["columns"]
    # latest.json is the same index: the whole table in one file passed 20 MB
    assert json.loads((api / "scores" / "latest.json").read_text())["shards"] == latest["shards"]
    board = json.loads((api / "scores" / "leaderboard.json").read_text())
    st_col = board["columns"].index("station_id")
    assert {r[st_col] for r in board["rows"]} == {ALL_STATIONS}


def test_scores_are_sharded_by_station_with_an_index(built):
    """One 21 MB file was both slow at the edge and near the 25 MiB Pages limit."""
    out, _, written, _ = built
    api = out / "api" / "v1"
    index = json.loads((api / "scores" / "index.json").read_text())
    assert index["n_rows"] > 0
    assert len(index["shards"]) == written["scores/by-station/{station}.json"]
    assert set(index["available"]["variables"]) >= {HEADLINE_VARIABLE, *SAMPLED_VARIABLES}
    total = 0
    for shard in index["shards"]:
        path = api / shard["path"]
        assert path.exists(), shard["path"]
        payload = json.loads(path.read_text())
        assert payload["station_id"] == shard["station_id"]
        # station_id is in the envelope, not repeated on every row
        assert "station_id" not in payload["columns"]
        assert len(payload["rows"]) == shard["rows"]
        total += len(payload["rows"])
        assert path.stat().st_size < 25 * 1024 * 1024
    assert total == index["n_rows"]


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
    for rel in ("scores/index.json", "scores/leaderboard.json", "pairwise/latest.json",
                "stations.json", "models.json",
                "leaderboard/90d-00z-bilinear-t2.json"):
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
    shard = json.loads(
        (out / "api" / "v1" / "scores" / "by-station" / "KAAA.json").read_text())
    i = shard["columns"].index("permalink")
    assert all(r[i].startswith("/station/KAAA/") and r[i].endswith("/") for r in shard["rows"][:50])
    board = json.loads(
        (out / "api" / "v1" / "leaderboard" / "90d-00z-bilinear-t2.json").read_text())
    assert board["view"] == {"window": "90d", "init_hour": 0, "method": "bilinear",
                            "variable": "t2",
                            "page": f"{SITE_URL}/v/90d-00z-bilinear-t2/"}
    assert board["results"]
    for r in board["results"]:
        assert r["permalink"].startswith("/station/ALL/model/")
        assert (r["rank"] is None) == (not r["ranked"])
        assert not r["ranked"] or r["n"] >= MIN_N


def test_one_leaderboard_file_per_site_view(built):
    out, _, _, _ = built
    files = sorted(p.name for p in (out / "api" / "v1" / "leaderboard").glob("*.json"))
    assert len(files) == len(LEADERBOARD_VIEWS) == 48
    assert "all-12z-nearest-tmin_s.json" in files
    assert "90d-00z-bilinear-t2.json" in files


def test_openapi_document_describes_the_endpoints(built):
    out, _, _, _ = built
    doc = json.loads((out / "api" / "v1" / "openapi.json").read_text())
    assert doc["openapi"].startswith("3.1")
    assert doc == openapi_document()
    for path in ("/scores/index.json", "/scores/latest.json",
                 "/scores/by-station/{station}.json", "/leaderboard/{view}.json",
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
        "v/90d-12z-bilinear-t2/index.html",
        "v/30d-00z-nearest-tmin_s/index.html",
        "v/all-12z-nearest-tmax_s/index.html",
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
        "data/scores_latest.csv.gz",
        "data/daily_errors/KAAA.csv.gz",
        "data/stations.csv",
        "data/models.csv",
        "assets/site.css",
        "assets/chart.js",
    ):
        assert (out / rel).exists(), rel
    assert counts["leaderboards"] == len(VIEWS) == 48
    assert len(SUBVIEWS) == 8
    assert counts["permalinks"] > 0
    assert counts["pages"] == counts["leaderboards"] + counts["permalinks"] \
        + len(SUBVIEWS) * counts["stations"] + len(SUBVIEWS) * counts["models"] + 6
    assert counts["files"] > counts["pages"]
    assert counts["bytes"] > 0


def test_every_leaderboard_view_is_a_static_page_with_working_switchers(built):
    out, _, _, _ = built
    for window, init_hour, method, variable in VIEWS:
        slug = view_slug(window, init_hour, method, variable)
        rel = "index.html" if slug == "90d-00z-bilinear-t2" else f"v/{slug}/index.html"
        html = (out / rel).read_text(encoding="utf-8")
        assert 'aria-current="page"' in html, rel
        # the switcher offers plain links, never a script-driven control
        assert 'href="/v/' in html or slug == "90d-00z-bilinear-t2"
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
    assert "Alpha Regional · warm1 family · lead day 1" in text_of(card)


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
    assert "Skill" in html and "Skill, debiased" in text_of(html)
    assert "Short" in text_of(html) and "Long" in text_of(html)  # the two citation forms
    assert 'rel="canonical"' in html
    body = text_of(html)
    # every window, interpolation and initialization of this combination is on the one page
    for window in ("30d", "90d", "all"):
        assert window in body, window
    assert "bilinear" in body
    assert re.search(r"\d\dZ", body)
    # compact interval notation, e.g. "[1.9, 2.3]" — a displayed negative is U+2212, not a hyphen,
    # so a column of signed numbers lines up in the monospace face
    assert re.search(r"\[−?\d+\.\d\d, −?\d+\.\d\d\]", body)
    assert not re.search(r'class="(?:val|ci)">\[?-\d', html), "a displayed number uses U+2212"


def test_permalink_csv_has_a_header_and_the_daily_errors(built):
    out, _, _, _ = built
    csv = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "errors.csv").read_text()
    lines = csv.strip().splitlines()
    assert lines[0].split(",")[-1] == "error_f"
    assert len(lines) > 1
    assert lines[1].startswith("KAAA,warm1,1,")


def test_leaderboard_ranks_by_mae_and_marks_low_n(dataset, tmp_path):
    daily, truth, instant, scores, pairwise = dataset
    build_site(as_of="2026-08-29", out=tmp_path, scores=scores, pairwise=pairwise,
               daily=daily, truth=truth, instant=instant, stations=STATION_OBJS,
               models=MODEL_OBJS, permalinks=False)
    html = (tmp_path / "index.html").read_text()
    assert "How far off was each weather model?" in html
    assert html.index("exact family") < html.index("cold18 family")
    assert "Data availability" in html
    assert f"n&nbsp;&lt;&nbsp;{MIN_N}" in html
    # MAE and bias are published side by side, each with its own interval
    assert "MAE °F" in html and "Bias °F" in html
    assert "Skill, debiased (out-of-sample)" in html
    # the leader is marked and the significance legend is present
    assert "★" in html


def test_headline_states_the_conclusion_with_n_window_and_update_times(dataset, tmp_path):
    daily, truth, instant, scores, pairwise = dataset
    build_site(as_of="2026-08-29", out=tmp_path, scores=scores, pairwise=pairwise,
               daily=daily, truth=truth, instant=instant, stations=STATION_OBJS,
               models=MODEL_OBJS, permalinks=False)
    html = (tmp_path / "index.html").read_text()
    assert "data through" in html
    assert "next update" in html
    assert "scored days" in html or "n &lt;" in html
    assert "window" in html and "interpolation" in html


def test_low_n_rows_are_greyed(tmp_path, dataset):
    daily, truth, instant, _, _ = dataset
    few = pd.to_datetime(daily["climo_date"]) >= pd.Timestamp("2026-08-20")
    scores, pairwise = score(daily[few], truth, windows=(None,), n_boot=100, seed=0)
    build_site(as_of="2026-08-29", out=tmp_path, scores=scores, pairwise=pairwise,
               daily=daily[few], truth=truth, stations=STATION_OBJS, models=MODEL_OBJS)
    html = (tmp_path / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    assert "is-lown" in html


def test_methodology_page_renders_the_document(built):
    out, _, _, _ = built
    html = (out / "methodology" / "index.html").read_text()
    assert "Climatological day" in html
    assert 'id="23-what-is-scored-v03"' in html  # anchored from the footer
    assert "<table" in html


def test_data_page_lists_schema_units_licences_downloads_and_limitations(built):
    out, _, _, _ = built
    html = (out / "data" / "index.html").read_text()
    assert "CC BY 4.0" in html and "public domain" in html
    assert "10.1175/BAMS-D-24-0057.1" in html  # the AIWP citation
    assert "daily_forecasts" in html and "pairwise" in html
    assert "Raw model output" in html
    assert "scores_latest.csv.gz" in html and "daily_errors/" in html
    assert "stations.csv" in html and "models.csv" in html
    assert "degC" in html or "°C" in html
    assert "Known limitations" in html and "Changelog" in html
    assert "<th scope=\"col\">Unit</th>" in html


def test_status_page_is_an_uptime_view(built):
    out, _, _, _ = built
    html = (out / "status" / "index.html").read_text()
    assert "Model runs" in html and "Truth (first-final NWS CLI)" in html
    assert 'class="cell ' in html or "cellbox" in html  # the per-day availability grid
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
        "models/index.html", "feed.xml", "_headers", "data/scores_latest.csv.gz"):
        assert (tmp_path / rel).exists(), rel


def test_export_api_without_data_produces_empty_endpoints(tmp_path):
    written = export_api(pd.DataFrame(), pd.DataFrame(), STATION_OBJS, MODEL_OBJS,
                         out=tmp_path / "api" / "v1")
    assert written["scores/by-station/{station}.json"] == 0
    payload = json.loads((tmp_path / "api" / "v1" / "scores" / "index.json").read_text())
    assert payload["rows"] == [] and payload["shards"] == []
    assert payload["window"]["type"] == "multiple"
    assert (tmp_path / "api" / "v1" / "leaderboard"
            / "90d-00z-bilinear-t2.json").exists()


# ------------------------------------------------------------------- methodology v0.3 (DESIGN §10)

def test_headline_is_the_instantaneous_variable_with_the_sampled_extremes_beside_it(built):
    """§10.2: t2 is the main table; the like-for-like extremes are the second one."""
    out, _, _, _ = built
    html = (out / "index.html").read_text()
    assert 'aria-current="page"' in html
    # the variable switcher offers exactly the three rankable variables
    for v in (HEADLINE_VARIABLE, *SAMPLED_VARIABLES):
        assert f"-{v}/" in html or f"bilinear-{v}" in html or v in html, v
    assert "instantaneous" in html
    assert "The same four samples as a daily maximum and minimum" in html
    # …and the CLI comparison is not on the front page at all
    assert "Against the NWS daily extremes" not in html


def test_cli_comparison_is_secondary_and_always_carries_its_caveat(built):
    """A2/§10.2: the sampling penalty is model-dependent, so those numbers are never ranked."""
    out, _, _, _ = built
    for rel in ("station/KAAA/index.html", "model/warm1/index.html",
                "station/KAAA/model/warm1/lead/1/index.html"):
        html = (out / rel).read_text()
        body = text_of(html)
        assert "Against the NWS daily extremes (secondary)" in body, rel
        assert "depends on each model's own diurnal amplitude" in body, rel
        assert "never used for ranking" in body, rel
    # the claim the review asked to be removed must not come back
    for page in out.rglob("index.html"):
        text = text_of(page.read_text())
        assert "identical for every model" not in text, page
        assert "affects all models equally" not in text, page


def test_skill_column_names_its_own_denominator_and_sample(built):
    """A1: the skill column and the persistence row must be reconcilable on the same page."""
    out, _, _, _ = built
    html = (out / "index.html").read_text()
    assert re.search(r"vs \d+\.\d\d \(n=\d+\)", html), "skill must print MAE(persistence) and n"
    assert "all days" in html, "the baseline row's own n is its whole record, and must say so"
    assert "on the days both have a value" in text_of(html)
    card = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    body = text_of(card)
    assert re.search(r"vs \d+\.\d\d \(n=\d+\)", body)
    assert "out of sample" in body and "n=" in body
    assert "Skill, debiased (out-of-sample)" in body


def test_only_holm_corrected_differences_are_marked(built):
    """B2: ▼/▲ follow distinguishable_holm; the uncorrected flag stays on the pairwise table."""
    out, _, _, _ = built
    html = (out / "index.html").read_text()
    assert "Holm-corrected within this table" in text_of(html)
    card = text_of((out / "station" / "ALL" / "model" / "warm1" / "lead" / "1"
                    / "index.html").read_text())
    assert "Distinguishable (Holm)" in card and "Distinguishable (uncorrected)" in card


def test_missing_v03_columns_degrade_to_dashes(dataset, tmp_path):
    """The site must build against a scores table written by an older methodology version."""
    daily, truth, instant, scores, pairwise = dataset
    old_scores = scores.drop(columns=["n_common", "mae_persistence_common", "skill_ci_low",
                                      "skill_ci_high", "n_debiased"])
    old_pw = pairwise.drop(columns=["distinguishable_holm", "distinguishable_uncorrected",
                                    "p_boot"])
    build_site(as_of="2026-08-29", out=tmp_path, scores=old_scores, pairwise=old_pw,
               daily=daily, truth=truth, instant=instant, stations=STATION_OBJS,
               models=MODEL_OBJS, permalinks=True)
    html = (tmp_path / "index.html").read_text()
    assert "—" in html
    assert not re.search(r"vs \d+\.\d\d \(n=\d+\)", html)
    # with no Holm verdict available nothing is declared distinguishable
    assert "▼" not in html and "▲" not in html
    assert "no Holm-corrected comparison is available" in text_of(html)
    card = (tmp_path / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    assert "Distinguishable (Holm)" in card


def test_pre_v03_variable_names_render_as_the_cli_comparison(dataset, tmp_path):
    """tmax/tmin were always the four samples against the NWS extreme: that is tmax_cli now."""
    daily, truth, instant, scores, pairwise = dataset
    legacy = scores[scores["variable"].isin(["t2", "tmin_s"])].copy()
    legacy["variable"] = legacy["variable"].map({"t2": "tmax", "tmin_s": "tmin"})
    build_site(as_of="2026-08-29", out=tmp_path, scores=legacy, pairwise=pd.DataFrame(),
               daily=daily, truth=truth, stations=STATION_OBJS, models=MODEL_OBJS)
    html = (tmp_path / "station" / "KAAA" / "index.html").read_text()
    assert "Against the NWS daily extremes (secondary)" in text_of(html)
    assert "tmax_cli" in html


def test_confidence_intervals_state_when_they_could_not_be_computed(built):
    out, _, _, _ = built
    html = (out / "index.html").read_text()
    assert "fewer than 28 scored days or fewer than 4 blocks" in text_of(html)
    assert "Wilson" in html


def test_station_directory_counts_forecast_systems_only(built):
    """A4: the persistence baseline is not a model, and was inflating every station's n."""
    out, _, _, dataset_report = built
    html = (out / "stations" / "index.html").read_text()
    models_html = (out / "models" / "index.html").read_text()
    assert 'href="/model/persistence/"' not in html
    # the last two numeric cells of each station row are (scored days, models)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    n_station = [int(m[-2]) for m in
                 (re.findall(r">(\d+)<", text_or_cells) for text_or_cells in rows)
                 if len(m) >= 2]
    assert n_station, "expected a scored-day count per station"
    n_model = [int(x) for x in re.findall(r">(\d+)<", models_html)]
    assert max(n_station) <= max(n_model), "a station cannot have more scored days than any model"


def test_stations_page_publishes_the_representativeness_terms(built):
    """B7: Δz and its lapse-rate magnitude, and the renamed market_city column."""
    out, _, _, _ = built
    html = (out / "stations" / "index.html").read_text()
    body = text_of(html)
    assert "Grid elev" in body and "Δz" in body and "6.5 K/km" in body
    assert "Market city" in body and "kalshi" not in html
    assert "not a random sample" in body
    csv = (out / "data" / "stations.csv").read_text()
    header = csv.splitlines()[0].split(",")
    assert header[:4] == ["station_id", "name", "cli_pil", "iem_id"]
    for col in ("grid_elev_m", "dz_m", "lapse_k", "market_city"):
        assert col in header, col
    assert "kalshi" not in header


def test_maps_are_a_fixed_reference_and_an_all_model_mean(built):
    """A6: never the per-station winner — on 28 days that map is a winner's-curse picture."""
    out, _, _, _ = built
    html = (out / "index.html").read_text()
    if "Where the errors are" in html:
        body = text_of(html)
        assert "best model" not in body and "Best model" not in body
        assert "All-model mean bias" in body
        assert "Neither shows which model" in body
        # the reference-model map is only drawn when that model is in the registry
        assert html.count("<figure") >= 1


def test_footer_carries_the_commit_and_a_citation_route(built):
    """C2: a published number can be tied to the code that produced it."""
    out, _, _, _ = built
    assert source_commit()  # a short hash in a checkout, "local" outside one
    html = (out / "index.html").read_text()
    # not compared against a freshly-read HEAD: the data pipeline commits while a build runs, so
    # the stamped hash is the one the build saw, which is the point of stamping it.
    assert re.search(r'/commit/([0-9a-f]{7,40}|local)"><code>', html)
    assert 'href="/data/#cite"' in html
    assert REPO_URL in html and "github.com/zifanzhang/castcheck" not in html
    cff = (out.parent / "CITATION.cff") if (out.parent / "CITATION.cff").exists() else None
    assert cff is None or "cff-version" in cff.read_text()


def test_data_page_documents_the_instantaneous_layer_and_the_revision_cost(built):
    """C1 and C4."""
    out, _, _, _ = built
    html = (out / "data" / "index.html").read_text()
    assert "forecast_values" in html and "truth_instant" in html
    assert "daily_errors/" in html
    assert "What the first-final policy costs" in html
    assert "CITATION.cff" in html


def test_changelog_is_rendered_from_the_methodology_document(built):
    """A5: one changelog, in the document the version number belongs to."""
    entries = changelog_entries("# X\n\n## Changelog\n\n- **0.3 (2026-08-31)** — headline redefined.\n"
                                "  - the sampled extremes move to a secondary block\n"
                                "- **0.2 (2026-08-30)** — review.\n")
    assert [e["version"] for e in entries] == ["0.3", "0.2"]
    assert entries[0]["details"] == ["the sampled extremes move to a secondary block"]
    out, _, _, _ = built
    html = (out / "data" / "index.html").read_text()
    for e in changelog_entries():
        assert f"v{e['version']}" in html, e["version"]


def test_no_page_attributes_the_cold_bias_to_a_cause(built):
    """A7/§10.6: the phenomenon is reported, never explained, until the analyses are done."""
    out, _, _, _ = built
    for page in out.rglob("*.html"):
        text = page.read_text()
        low = text.lower()
        assert "era5" not in low or "not yet attributed" in low, page
        for phrase in ("because ERA5", "due to ERA5", "caused by ERA5", "ERA5 training explains"):
            assert phrase not in text, page


def test_status_counts_uptime_from_each_models_period_start(built):
    """B8: the days before a model existed are not downtime."""
    _, _, _, report = built
    assert "basis" in report["uptime"]
    for m in report["models"]:
        assert "period_start" in m and "n_expected" in m
        for d in m["days"]:
            assert "expected" in d and "reason" in d
    body = text_of((out_of(built) / "status" / "index.html").read_text())
    assert "not produced upstream" in body
    assert "from each model's own period start" in body


def out_of(built):
    return built[0]


def test_status_expected_steps_exclude_the_analysis_step():
    """AIWP files carry f000; counting rows would pass a run that is missing f240."""
    values = pd.DataFrame([
        {"model_id": "m", "station_id": "KAAA", "init_time": pd.Timestamp("2026-08-30", tz="UTC"),
         "valid_time": pd.Timestamp("2026-08-30", tz="UTC") + pd.Timedelta(hours=h),
         "lead_h": h, "variable": "t2", "value_c": 1.0, "missing_reason": "", "method": "bilinear"}
        for h in range(0, 13, 6)  # f000, f006, f012 — two forecast steps, three rows
    ])
    model = ModelSpec(model_id="m", family="m", source="aiwp", product="p", init_field="GFS",
                      inits=(0,), step_h=6, max_h=12, native_extremes=())
    report = status_mod.build(as_of="2026-08-30", days=1, values=values,
                              truth=pd.DataFrame(), stations=STATION_OBJS[:1], models=[model],
                              upstream=False)
    assert report["models"][0]["expected_steps"] == 2
    assert report["models"][0]["days"][0]["complete"] is True
    short = values[values["lead_h"] != 12]
    report = status_mod.build(as_of="2026-08-30", days=1, values=short,
                              truth=pd.DataFrame(), stations=STATION_OBJS[:1], models=[model],
                              upstream=False)
    assert report["models"][0]["days"][0]["complete"] is False


def test_status_reports_observed_instant_coverage():
    """The truth of the headline score needs its own coverage bar: four instants a day."""
    values = pd.DataFrame([
        {"model_id": "exact", "station_id": "KAAA",
         "init_time": pd.Timestamp("2026-08-29", tz="UTC"),
         "valid_time": pd.Timestamp("2026-08-29", tz="UTC"), "lead_h": 6, "variable": "t2",
         "value_c": 1.0, "missing_reason": "", "method": "bilinear"}])
    ti = pd.DataFrame([
        {"station_id": "KAAA",
         "valid_time": pd.Timestamp("2026-08-29", tz="UTC") + pd.Timedelta(hours=h),
         "temp_c": 20.0}
        for h in (0, 6, 12)  # one instant short of a complete day
    ])
    report = status_mod.build(as_of="2026-08-30", days=2, values=values, truth=pd.DataFrame(),
                              truth_instant=ti, stations=STATION_OBJS[:1], models=MODEL_OBJS[:1],
                              upstream=False)
    row = report["truth_instant"][0]
    assert row["station_id"] == "KAAA" and row["period_start"] == "2026-08-29"
    day = next(d for d in row["days"] if d["date"] == "2026-08-29")
    assert day["n_instants"] == 3 and day["complete"] is False and day["expected"] is True
    # today's 18Z has not happened yet, so today is never a hole
    today = next(d for d in row["days"] if d["date"] == "2026-08-30")
    assert today["expected"] is False and today["reason"] == "not_due_yet"
    assert report["uptime"]["truth_instant"] == 0.0
    assert any(g["type"] == "truth_instant" for g in report["gaps"])


def test_status_page_shows_the_instant_coverage_section(built):
    out, _, _, report = built
    html = (out / "status" / "index.html").read_text()
    if report.get("truth_instant"):
        assert "Observed instants" in html
        assert "truth_instant" in html


def test_daily_errors_are_sharded_per_station_and_name_the_instant(built):
    """The combined file reached the 25 MB Cloudflare Pages limit once t2 added four rows a day."""
    out, _, _, _ = built
    assert not (out / "data" / "daily_errors.csv.gz").exists()
    shards = sorted((out / "data" / "daily_errors").glob("*.csv.gz"))
    assert shards
    for path in shards:
        assert path.stat().st_size < 25 * 1024 * 1024, path
    import gzip as _gzip
    with _gzip.open(out / "data" / "daily_errors" / "KAAA.csv.gz", "rt") as f:
        header = f.readline().strip().split(",")
        rows = [f.readline().strip().split(",") for _ in range(3)]
    assert "valid_hour_utc" in header, "the four t2 rows of a day must be distinguishable"
    assert header[-1] == "err"
    assert all(r[header.index("station_id")] == "KAAA" for r in rows if r and r[0])


# ------------------------------------------------------------------------------------------
# the 2026-08 visual redesign: the component vocabulary the stylesheet is written against
# ------------------------------------------------------------------------------------------

def test_every_page_loads_the_three_typefaces_with_a_fallback_stack(built):
    """Source Serif 4 / Inter / IBM Plex Mono, preconnected and swap-rendered."""
    out, _, _, _ = built
    html = (out / "index.html").read_text(encoding="utf-8")
    assert '<link rel="preconnect" href="https://fonts.googleapis.com">' in html
    assert 'href="https://fonts.gstatic.com" crossorigin' in html
    link = re.search(r'<link rel="stylesheet" href="(https://fonts\.googleapis\.com/css2[^"]+)"',
                     html)
    assert link, "the Google Fonts stylesheet must be linked"
    for family in ("Source+Serif+4", "Inter", "IBM+Plex+Mono"):
        assert family in link.group(1), family
    assert "display=swap" in link.group(1)
    css = (out / "assets" / "site.css").read_text(encoding="utf-8")
    # a web font that fails to load must never leave the page without a face
    assert '--font-display: "Source Serif 4", Georgia' in css
    assert '--font-sans: "Inter", system-ui' in css
    assert '--font-mono: "IBM Plex Mono", ui-monospace' in css


def test_the_component_vocabulary_is_defined_and_used(built):
    """Every class the templates emit has a rule; the stylesheet has no orphan components."""
    out, _, _, _ = built
    css = (out / "assets" / "site.css").read_text(encoding="utf-8")
    for selector in (".panel", "table.data", ".filterbar", ".segmented", ".chip",
                     ".metastrip", ".kpi-row", ".fig__title", ".fig__source", ".avail__track",
                     ".uptime-grid", ".statusbar", ".cite__row", ".site-header",
                     ".site-footer__grid", ".model-grid", ".mcard", ".badge", ".downloads",
                     ".empty-state", "details.behind", ".prose"):
        assert selector in css, selector
    for token in ("--paper:", "--ink-1:", "--link:", "--warm-2:", "--cool-2:", "--sp-4:"):
        assert token in css, token
    home = (out / "index.html").read_text(encoding="utf-8")
    for cls in ('class="site-header"', 'class="hero__title"', 'class="metastrip"',
                'class="kpi-row"', 'class="filterbar"', 'class="panel"', 'class="data"',
                'class="site-footer__grid"'):
        assert cls in home, cls


def test_home_page_leads_with_a_verdict_kpis_and_an_error_bar_chart(built):
    out, _, _, _ = built
    html = (out / "index.html").read_text(encoding="utf-8")
    assert 'class="hero__verdict"' in html
    # four KPI cards, each a label / figure / footnote triple
    assert html.count('class="kpi__label"') == 4
    assert html.count('class="kpi__value"') == 4
    body = text_of(html)
    for label in ("Leader, lead day", "Persistence baseline", "Scored days", "Systems ranked"):
        assert label in body, label
    # the ranked bars with whiskers, drawn server-side
    assert 'class="c-bar' in html and 'class="c-ci"' in html
    assert "Mean absolute error with 95 % confidence intervals" in body
    # one panel per lead day, each with its own permalink anchor
    assert 'id="lead1"' in html


def test_models_page_is_a_card_grid_over_the_same_table(built):
    out, _, _, _ = built
    html = (out / "models" / "index.html").read_text(encoding="utf-8")
    assert 'class="model-grid"' in html
    assert html.count('class="mcard') >= len(MODEL_OBJS)
    for m in MODEL_OBJS:
        assert f'<span class="mcard__id">{m.model_id}</span>' in html
    assert 'class="avail__track"' in html and 'class="avail__scale"' in html
    assert 'class="mcard__spark"' in html
    body = text_of(html)
    assert "scored day" in body and "Model page" in body
    # the full table is still published under the cards, with every column
    assert 'class="data"' in html
    for column in ("Source", "Initial conditions", "Version segment", "Period of record"):
        assert column in body, column
    assert "The same list as a table" in body


def test_status_page_opens_with_a_status_bar_and_kpis(built):
    out, _, _, _ = built
    html = (out / "status" / "index.html").read_text(encoding="utf-8")
    assert re.search(r'class="statusbar is-(ok|warn|bad)"', html)
    assert 'class="statusbar__dot"' in html and 'class="statusbar__meta"' in html
    assert html.count('class="kpi__value"') == 4
    assert 'class="data uptime-grid"' in html and 'class="cell ' in html
    assert re.search(r'class="badge badge--(ok|warn|bad)"', html)


def test_the_pages_are_complete_without_javascript(built):
    """No JS: the navigation, the tables and the view switcher are all plain markup."""
    out, _, _, _ = built
    for rel in ("index.html", "models/index.html", "stations/index.html", "status/index.html",
                "station/KAAA/model/warm1/lead/1/index.html"):
        html = (out / rel).read_text(encoding="utf-8")
        # nothing is drawn by a script: no inline handlers, no <template>, no empty mount point
        assert "onclick" not in html and "<template" not in html, rel
        assert "document.write" not in html, rel
        # the navigation is seven ordinary links
        for href in ('href="/"', 'href="/stations/"', 'href="/models/"', 'href="/methodology/"',
                     'href="/status/"', 'href="/data/"', 'href="/api/v1/"'):
            assert href in html, f"{rel}: {href}"
        # the numbers are in the served HTML, inside a real table
        assert "<table" in html and "<tbody" in html, rel
        # the only script tags are the theme pre-paint inline one and the deferred enhancement
        assert html.count("<script") == 2, rel
        assert "chart." in html and "defer" in html, rel
    # the view switcher is a set of links, never a form control
    home = (out / "index.html").read_text(encoding="utf-8")
    assert "<select" not in home and "<button" not in home.replace(
        '<button class="btn btn--sm theme-toggle"', "")
    assert 'class="chip is-active"' in home and 'aria-current="page"' in home


def test_the_permalink_collapses_the_view_key_and_marks_the_current_row(built):
    out, _, _, _ = built
    html = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1" / "index.html").read_text()
    # window · init · method as one monospace key instead of four columns
    assert re.search(r'class="keycell">\s*90d · \d\dZ · \w+', html)
    assert 'class="is-current"' in html and "this view" in html
    assert 'class="rowgroup"' in html          # grouped by variable
    assert 'class="chart-pair"' in html        # series and histogram side by side
    assert 'class="cite__row"' in html and 'class="cite__k"' in html
    assert 'class="downloads"' in html and 'class="fmt"' in html


def test_the_home_chart_draws_bars_and_whiskers_in_the_same_unit(built):
    """The bar length and its whiskers must come from one conversion, not two."""
    from castcheck.site.build import C_TO_F_DELTA, _row_view

    out, _, _, _ = built
    row = {"mae": 2.0, "mae_ci_low": 1.8, "mae_ci_high": 2.2, "bias": 0.5,
           "bias_ci_low": 0.3, "bias_ci_high": 0.7, "rmse": 2.5, "hit1f": 0.3, "hit2f": 0.5,
           "hit3f": 0.7, "skill_persistence": 0.4, "n": 50, "variable": HEADLINE_VARIABLE,
           "init_hour": 0, "method": "bilinear", "window": "90d",
           "period_start": "2026-01-01", "period_end": "2026-03-01"}
    v = _row_view(pd.Series(row), "KAAA", "warm1", 1)
    assert v["mae_f"] == pytest.approx(2.0 * C_TO_F_DELTA)
    assert v["mae"] == f"{v['mae_f']:.2f}"
    assert v["mae_ci_low_f"] < v["mae_f"] < v["mae_ci_high_f"]
    assert v["mae_ci_low_f"] == pytest.approx(1.8 * C_TO_F_DELTA)
    # and the axis of the published chart is consistent with the numbers in its own table
    html = (out / "index.html").read_text(encoding="utf-8")
    axis = max(float(x) for x in re.findall(r'class="c-tick">([\d.]+)</text>', html))
    printed = [float(x) for x in re.findall(r'class="c-val">([\d.]+)</text>', html)]
    assert printed and max(printed) <= axis
