"""Tests for the JSON export, the status report and the static site generator (DESIGN §6)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
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


#: A fixed "now" for every status report in this module.  ``status.build`` asks
#: ``castcheck.schedule`` whether each run is *due*, so a report built against the real clock
#: would change its answers as the day goes on.
NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


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
                              stations=STATION_OBJS, models=MODEL_OBJS, upstream=False,
                              now=NOW)
    written = export_api(scores, pairwise, STATION_OBJS, MODEL_OBJS, out=out / "api" / "v1",
                         daily=daily, truth=truth, instant=instant, status=report)
    counts = build_site(as_of="2026-08-29", out=out, scores=scores, pairwise=pairwise,
                        daily=daily, truth=truth, instant=instant, stations=STATION_OBJS,
                        models=MODEL_OBJS, status_report=report, api_written=written,
                        truth_instant=make_truth_instant())
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
    # station_id is the shard key, so it is in the envelope, not in the column list
    assert "station_id" not in latest["columns"] and "mae_ci_low" in latest["columns"]
    assert latest["permalink_template"].endswith("/model/{model_id}/lead/{lead_day}/")
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
    assert len(index["shards"]) == written["station/{station}/cards.json"]
    assert set(index["available"]["variables"]) >= {HEADLINE_VARIABLE, *SAMPLED_VARIABLES}
    total = 0
    for shard in index["shards"]:
        path = out / shard["path"]
        assert path.exists(), shard["path"]
        payload = json.loads(path.read_text())
        assert payload["station_id"] == shard["station_id"]
        # station_id is in the envelope, not repeated on every row
        assert "station_id" not in payload["scores"]["columns"]
        assert len(payload["scores"]["rows"]) == shard["rows"]
        total += len(payload["scores"]["rows"])
        assert path.stat().st_size < 25 * 1024 * 1024
        # the historical shard path still resolves and says where the rows went
        stub = json.loads((api / "scores" / "by-station"
                           / f"{shard['station_id']}.json").read_text())
        assert stub["moved_to"] == shard["href"] and stub["n_rows"] == shard["rows"]
    assert total == index["n_rows"]


def test_the_card_bundle_is_one_file_per_station(built):
    """1 848 card files repeated the envelope, the units block and the whole scores table."""
    out, _, written, _ = built
    bundle = json.loads((out / "station" / "KAAA" / "cards.json").read_text())
    assert bundle["kind"] == "station-cards" and bundle["station_id"] == "KAAA"
    assert bundle["n_scores"] == len(bundle["scores"]["rows"]) > 0
    by_id = {c["id"]: c for c in bundle["cards"]}
    card = by_id["warm1-1"]
    assert card["permalink"] == "/station/KAAA/model/warm1/lead/1/"
    assert len(card["pairwise"]["rows"]) > 0
    # the constants of the card's pairwise slice are in pairwise_scope, not on every row
    assert not ({"station_id", "lead_day", "window", "init_hour", "method"}
                & set(card["pairwise"]["columns"]))
    assert bundle["pairwise_scope"]["window"] and bundle["pairwise_scope"]["method"]
    # the eleven variables share one date axis instead of repeating it eleven times
    assert card["series"], "expected a daily error series"
    assert card["series_dates"]
    assert all("dates" not in e for e in card["series"])
    assert len(card["series"][0]["err_c"]) == len(card["series_dates"])
    # and the page links to its own card by fragment
    html = (out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1"
            / "index.html").read_text()
    assert "/station/KAAA/cards.json#warm1-1" in html
    # the per-card files are gone
    assert not (out / "api" / "v1" / "scores" / "KAAA").exists()


def test_the_bundle_dictionary_encodes_its_label_columns(built):
    """Eight short labels repeated on every row were 39 % of the file."""
    out, _, _, _ = built
    scores = json.loads((out / "station" / "KAAA" / "cards.json").read_text())["scores"]
    dicts = scores["dictionaries"]
    assert set(dicts) >= {"model_id", "variable", "method", "window"}
    for col, values in dicts.items():
        i = scores["columns"].index(col)
        codes = {r[i] for r in scores["rows"]}
        assert all(isinstance(c, int) for c in codes)
        assert codes <= set(range(len(values)))
    # and it round-trips to the values the plain endpoints publish
    i = scores["columns"].index("variable")
    decoded = {dicts["variable"][r[i]] for r in scores["rows"]}
    assert HEADLINE_VARIABLE in decoded


def test_permalink_json_has_scores_pairwise_and_series(built):
    out, _, _, _ = built
    bundle = json.loads((out / "station" / "KAAA" / "cards.json").read_text())
    assert bundle["station_id"] == "KAAA"
    card = next(c for c in bundle["cards"] if c["id"] == "warm1-1")
    assert card["model_id"] == "warm1" and card["lead_day"] == 1
    assert card["permalink"] == "/station/KAAA/model/warm1/lead/1/"
    cols = bundle["scores"]["columns"]
    mi, li = cols.index("model_id"), cols.index("lead_day")
    vocab = bundle["scores"]["dictionaries"]["model_id"]
    assert [r for r in bundle["scores"]["rows"]
            if vocab[r[mi]] == "warm1" and r[li] == 1]
    assert len(card["pairwise"]["rows"]) > 0
    assert card["series"], "expected a daily error series"
    e = card["series"][0]
    dates = e.get("dates") or card["series_dates"]
    assert len(dates) == len(e["err_c"]) <= bundle["series_days"]


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
    """The leaderboard slices, which are what a casual consumer reads, still carry a literal
    permalink; the bulk per-station table carries model_id + lead_day and the template."""
    out, _, _, _ = built
    shard = json.loads((out / "api" / "v1" / "scores" / "leaderboard.json").read_text())
    i = shard["columns"].index("permalink")
    assert all(r[i].startswith(f"/station/{ALL_STATIONS}/") and r[i].endswith("/")
               for r in shard["rows"][:50])
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
                 "/station/{station}/cards.json", "/status.json"):
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
    # the always-present pages: /, methodology, 404, status, stations, models, api, data,
    # diagnostics, monthly index — plus one page per completed month
    assert counts["pages"] == counts["leaderboards"] + counts["permalinks"] \
        + len(SUBVIEWS) * counts["stations"] + len(SUBVIEWS) * counts["models"] \
        + counts["months"] + 9
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
    assert "/station/KAAA/cards.json#warm1-1" in html
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
                        models=MODEL_OBJS, truth_instant=pd.DataFrame())
    # …/diagnostics/ and …/monthly/ are in the navigation from the first build, so they are
    # written with an empty state rather than skipped into a 404
    assert counts["pages"] == 10
    assert counts["months"] == 0
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
                              upstream=False, now=NOW)
    assert report["models"][0]["expected_steps"] == 2
    assert report["models"][0]["days"][0]["complete"] is True
    short = values[values["lead_h"] != 12]
    report = status_mod.build(as_of="2026-08-30", days=1, values=short,
                              truth=pd.DataFrame(), stations=STATION_OBJS[:1], models=[model],
                              upstream=False, now=NOW)
    assert report["models"][0]["days"][0]["complete"] is False


def test_minify_html_drops_indentation_without_changing_the_rendering():
    """The generator's indentation was a tenth of the bytes of ~4 000 files."""
    from castcheck.site.build import minify_html

    out = minify_html(
        "<div>\n  <p>hello\n     world</p>\n  <ul>\n    <li><a href=\"/a\">A</a>\n"
        "    <a href=\"/b\">B</a></li>\n  </ul>\n  <table><tbody>\n    <tr>\n"
        "      <td>1</td>\n      <td>2</td>\n    </tr>\n  </tbody></table>\n</div>\n"
        "<pre>\n  keep   me\n</pre>")
    # between block-level tags the whitespace is dropped …
    assert "<div><p>hello world</p><ul><li>" in out
    assert "<tr><td>1</td><td>2</td></tr>" in out
    # … but between two inline elements it is the space between two words
    assert '<a href="/a">A</a> <a href="/b">B</a>' in out
    # and preformatted text is untouched
    assert "<pre>\n  keep   me\n</pre>" in out


def test_no_page_scrolls_sideways_on_a_phone():
    """The visually-hidden spans inside a wide table are absolutely positioned at their static
    position — a metre off the right edge of the scroller. Without a positioned ancestor their
    containing block is the page, and the whole document scrolled sideways at 375 px."""
    css = (Path(__file__).resolve().parents[1] / "castcheck" / "site" / "assets"
           / "site.css").read_text(encoding="utf-8")
    rule = next(line for line in css.splitlines() if line.startswith(".table-wrap {"))
    assert "overflow-x: auto" in rule and "position: relative" in rule


def test_skill_is_blank_below_the_minimum_common_sample():
    """A ratio of two means over a handful of shared days is not a number worth printing."""
    from castcheck.api import SKILL_MIN_COMMON
    from castcheck.site.build import _row_view

    base = {"mae": 2.0, "mae_ci_low": None, "mae_ci_high": None, "bias": 0.5,
            "bias_ci_low": None, "bias_ci_high": None, "rmse": 2.5, "hit1f": 0.3, "hit2f": 0.5,
            "hit3f": 0.7, "skill_persistence": 0.42, "skill_persistence_debiased": 0.4,
            "skill_ci_low": 0.1, "skill_ci_high": 0.7, "n": 40, "variable": "t2",
            "init_hour": 0, "method": "bilinear", "window": "90d",
            "period_start": "2026-01-01", "period_end": "2026-02-01",
            "mae_persistence_common": 3.0}
    plenty = _row_view({**base, "n_common": SKILL_MIN_COMMON}, "KAAA", "gfs", 1)
    assert plenty["skill_reliable"] is True and plenty["skill"] != "—"
    assert plenty["skill_f"] is not None

    scarce = _row_view({**base, "n_common": SKILL_MIN_COMMON - 1}, "KAAA", "gfs", 1)
    assert scarce["skill_reliable"] is False
    assert scarce["skill"] == "—" and scarce["skill_ci"] == "—"
    assert scarce["skill_debiased"] == "—" and scarce["skill_f"] is None
    # the sample size stays visible, so the reader sees why
    assert f"n={SKILL_MIN_COMMON - 1}" in scarce["skill_vs"]


def test_timestamps_are_human_with_a_machine_readable_attribute(built):
    from castcheck.site.build import human_time

    assert human_time("2026-08-31T05:46:12+00:00") == "2026-08-31 05:46 UTC"
    assert human_time(None) == "—"
    out, _, _, _ = built
    html = (out / "index.html").read_text()
    assert re.search(r'<time datetime="20[^"]+">\d{4}-\d\d-\d\d \d\d:\d\d UTC</time>', html)
    assert "last build <time datetime=" in html


def test_status_marks_a_run_that_is_not_due_yet_instead_of_missing():
    """A 12Z ECMWF run does not exist at 13 UTC; the status page must not call that a gap."""
    model = ModelSpec(model_id="m", family="m", source="ecmwf", product="oper", init_field=None,
                      inits=(0, 12), step_h=6, max_h=12, native_extremes=())
    values = pd.DataFrame([
        {"model_id": "m", "station_id": "KAAA", "init_time": pd.Timestamp("2026-08-30", tz="UTC"),
         "valid_time": pd.Timestamp("2026-08-30", tz="UTC") + pd.Timedelta(hours=h),
         "lead_h": h, "variable": "t2", "value_c": 1.0, "missing_reason": "", "method": "bilinear"}
        for h in (6, 12)
    ])
    # 13:00 UTC: the 00Z run is eight hours old and complete, the 12Z run is one hour old.
    report = status_mod.build(as_of="2026-08-30", days=1, values=values, truth=pd.DataFrame(),
                              stations=STATION_OBJS[:1], models=[model], upstream=False,
                              now=datetime(2026, 8, 30, 13, 0, tzinfo=UTC))
    by_init = {m["init_hour"]: m for m in report["models"]}
    assert by_init[0]["days"][0]["complete"] is True
    late = by_init[12]["days"][0]
    assert late["complete"] is False
    assert late["expected"] is False and late["reason"] == "not_due_yet"
    assert late["due_at"] == "2026-08-30T20:00:00+00:00"  # 12Z + 8 h (ECMWF)
    assert by_init[12]["n_not_due_yet"] == 1
    # not a gap, so the CLI stays green, and the page says so rather than "34 items missing"
    assert report["n_current_gaps"] == 0 and report["ok"] is True
    assert any(p["type"] == "model_run" and p["init_hour"] == 12
               for p in report["pending"])
    assert status_mod.exit_code(report) == status_mod.EXIT_OK
    assert report["uptime"]["model_runs"] == 100.0

    # eight hours later the same absence *is* a gap
    later = status_mod.build(as_of="2026-08-30", days=1, values=values, truth=pd.DataFrame(),
                             stations=STATION_OBJS[:1], models=[model], upstream=False,
                             now=datetime(2026, 8, 30, 21, 0, tzinfo=UTC))
    assert later["n_current_gaps"] == 1 and later["ok"] is False
    assert status_mod.exit_code(later) == status_mod.EXIT_GAPS


def test_status_truth_deadline_is_per_station_local_midnight():
    """Yesterday's CLI exists in New York at 06 UTC and cannot exist yet in Chicago."""
    east = Station(id="KEEE", name="East", cli_pil="CLIEEE", tz="America/New_York",
                   std_offset_h=-5, lat=40.0, lon=-74.0, elev_m=10.0)
    west = Station(id="KWWW", name="West", cli_pil="CLIWWW", tz="America/Los_Angeles",
                   std_offset_h=-8, lat=34.0, lon=-118.0, elev_m=100.0)
    # 2026-08-30 10:00 UTC = 05:00 EST (four hours past midnight EST, so the report is due)
    # and 02:00 PST (two hours past midnight PST, so it is not).
    now = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
    report = status_mod.build(as_of="2026-08-30", days=2, values=pd.DataFrame(),
                              truth=pd.DataFrame(), truth_instant=pd.DataFrame(),
                              stations=[east, west], models=[], upstream=False, now=now)
    rows = {t["station_id"]: {d["date"]: d for d in t["days"]} for t in report["truth"]}
    assert rows["KEEE"]["2026-08-29"]["expected"] is True
    assert rows["KWWW"]["2026-08-29"]["expected"] is False
    assert rows["KWWW"]["2026-08-29"]["reason"] == "not_due_yet"
    # 2026-08-29 + 1 day at 08:00 UTC (00:00 PST) + 4 h
    assert rows["KWWW"]["2026-08-29"]["due_at"] == "2026-08-30T12:00:00+00:00"
    gaps = {(g["station_id"], g["date"]) for g in report["gaps"] if g["type"] == "truth"}
    assert ("KEEE", "2026-08-29") in gaps and ("KWWW", "2026-08-29") not in gaps


def test_status_page_says_all_due_runs_present_when_nothing_is_late(tmp_path):
    """The red "N item(s) missing" bar must not fire on runs that are merely not out yet."""
    from castcheck.site.build import _status_view

    report = {"days": 1, "ok": True, "n_current_gaps": 0, "n_gaps": 0, "n_pending": 3,
              "models": [], "truth": [], "truth_instant": [], "dates": ["2026-08-30"]}
    view = _status_view(report, names={})
    assert view["overall"] == "ok"
    assert "All due runs present" in view["overall_text"]
    assert "3 item(s) for the current day are not due yet" in view["overall_text"]
    assert _status_view({**report, "n_pending": 0}, names={})["overall_text"].startswith(
        "All systems operational")


def test_schedule_is_the_single_source_of_the_availability_delays():
    """cli and status must read the same numbers or the page contradicts the fetcher."""
    from castcheck import cli as cli_mod
    from castcheck import schedule

    assert cli_mod.AVAILABILITY_DELAY_H is schedule.AVAILABILITY_DELAY_H
    assert cli_mod.AIWP_DELAY_H_BY_INIT_FIELD is schedule.AIWP_DELAY_H_BY_INIT_FIELD
    aiwp_ifs = ModelSpec(model_id="a", family="a", source="aiwp", product="p", init_field="IFS",
                         inits=(0,), step_h=6, max_h=12, native_extremes=())
    aiwp_gfs = ModelSpec(model_id="b", family="b", source="aiwp", product="p", init_field="GFS",
                         inits=(0,), step_h=6, max_h=12, native_extremes=())
    assert cli_mod.availability_delay_h(aiwp_ifs) == 9.5
    assert cli_mod.availability_delay_h(aiwp_gfs) == 6.0
    init = datetime(2026, 8, 30, 0, 0, tzinfo=UTC)
    assert schedule.run_due_at(aiwp_gfs, init) == datetime(2026, 8, 30, 6, 0, tzinfo=UTC)
    assert schedule.run_is_due(aiwp_gfs, init, datetime(2026, 8, 30, 5, 0, tzinfo=UTC)) is False
    assert schedule.run_is_due(aiwp_gfs, init, datetime(2026, 8, 30, 6, 0, tzinfo=UTC)) is True


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
                              upstream=False,
                              now=datetime(2026, 8, 30, 12, 0, tzinfo=UTC))
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
    # window · init · method as one monospace key instead of four columns.  The compact `k`
    # class is the permanent-link table's spelling of `keycell` (same rendering, a third of the
    # bytes across ~1800 pages), and windows covering the same days share a row.
    assert re.search(r'class=k>[^<]*90d[^<]*· \d\dZ · \w+', html)
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


def test_404_page_is_generated(tmp_path):
    from castcheck.site.build import build_site

    build_site(as_of="2026-08-31", out=tmp_path, scores=None, permalinks=False)
    html = (tmp_path / "404.html").read_text(encoding="utf-8")
    assert "Page not found" in html and "/stations/" in html
    # the body is markup, not text: it must not arrive double-escaped
    assert '<a href="/stations/">stations</a>' in html
    assert "&lt;p&gt;" not in html


# ------------------------------------------------------- /diagnostics/ (METHODOLOGY §10.2)

def test_diagnostics_page_publishes_the_three_cuts(built):
    out, _, _, _ = built
    html = (out / "diagnostics" / "index.html").read_text(encoding="utf-8")
    body = text_of(html)
    # the three figures the page exists for
    assert 'id="by-hour"' in html and 'id="by-lead"' in html and 'id="sampling-penalty"' in html
    # the four synoptic instants, named as variables and as local time
    for var in ("t2_00z", "t2_18z"):
        assert var in body, var
    assert "Local standard time" in body
    # the sampled/CLI pair and the difference between them
    assert "tmax_s" in body and "tmax_cli" in body and "Penalty" in body


def test_diagnostics_reports_the_phenomenon_without_attributing_it(built):
    """A7/§10.2: three candidates, no cause, and a link to the document that says so in full."""
    out, _, _, _ = built
    html = (out / "diagnostics" / "index.html").read_text(encoding="utf-8")
    body = text_of(html)
    assert "not yet attributed" in body
    for candidate in ("Model diurnal amplitude", "The extreme-sampling penalty",
                      "The initial conditions"):
        assert candidate in body, candidate
    assert "#102-an-unattributed-empirical-observation" in html
    # nothing on the page claims a mechanism
    for phrase in ("because", "caused by", "explains", "due to the"):
        assert f" {phrase} the AI models" not in body.lower(), phrase
    # and it says the numbers move, so a reader does not quote them as settled
    assert "rebuilt from scratch on each publication" in body
    assert "Next rebuild" in body


def test_diagnostics_figures_are_accessible_and_have_an_equivalent_table(built):
    out, _, _, _ = built
    html = (out / "diagnostics" / "index.html").read_text(encoding="utf-8")
    charts = [c for c in html.split("<svg")[1:] if 'class="chart"' in c[:60]]
    assert len(charts) >= 2, "the hour figure and the lead figure are both server-rendered"
    for chunk in charts:
        assert 'role="img"' in chunk[:400]
        assert 'aria-label="' in chunk[:400]
        assert "<title>" in chunk[:600]
    # every figure has a table beside it, and the colour is never the only encoding
    assert html.count("<details") >= 2
    assert body_has_table(html, "The same figure as a table")
    assert "labelled at its right-hand end" in text_of(html)


def body_has_table(html: str, summary: str) -> bool:
    return summary in text_of(html) and "<table" in html


def test_diagnostics_json_is_the_same_numbers_as_the_page(built):
    out, _, written, _ = built
    payload = json.loads((out / "api" / "v1" / "diagnostics.json").read_text())
    assert written["diagnostics.json"] == 1
    for key in ("schema_version", "generated_at", "window", "units", "method", "truth"):
        assert key in payload, key
    assert payload["hourly_bias"]["hours_utc"] == [0, 6, 12, 18]
    assert payload["hourly_bias"]["variables"] == ["t2_00z", "t2_06z", "t2_12z", "t2_18z"]
    assert payload["bias_by_lead"]["variable"] == HEADLINE_VARIABLE
    ids = {r["model_id"] for r in payload["hourly_bias"]["rows"]}
    assert {m.model_id for m in MODEL_OBJS} <= ids
    # the fixture's `cold18` model is 3 °C cold at 18Z and exact elsewhere: the cut has to show it
    row = next(r for r in payload["hourly_bias"]["rows"] if r["model_id"] == "cold18")
    at = dict(zip(payload["hourly_bias"]["hours_utc"], row["bias"], strict=True))
    assert at[18] < -2.5 and abs(at[6]) < 0.5
    # the penalty is published with the sample size of both sides, never as a bare number
    pen = payload["sampling_penalty"]["rows"][0]
    assert "tmax_s_penalty" in pen and "tmax_s_n" in pen and "tmax_cli_n" in pen
    assert "attribution" in payload
    assert "is attributed to a cause" in payload["attribution"]


def test_diagnostics_openapi_describes_the_endpoint():
    doc = openapi_document()
    assert "/diagnostics.json" in doc["paths"]
    assert "Diagnostics" in doc["components"]["schemas"]


# ------------------------------------------------------------------- /monthly/

MONTHS_IN_FIXTURE = ("2026-06", "2026-07")


def test_monthly_pages_exist_only_for_completed_months(built):
    """The month in progress has no page: it would rank models and then change its own numbers."""
    out, counts, _, _ = built
    assert (out / "monthly" / "index.html").exists()
    for month in MONTHS_IN_FIXTURE:
        assert (out / "monthly" / month / "index.html").exists(), month
    # data runs to 2026-08-29, so August is not over and gets no page
    assert not (out / "monthly" / "2026-08").exists()
    assert counts["months"] == len(MONTHS_IN_FIXTURE)
    index = (out / "monthly" / "index.html").read_text()
    for month in MONTHS_IN_FIXTURE:
        assert f'href="/monthly/{month}/"' in index, month
    assert 'href="/monthly/2026-08/"' not in index


def test_a_monthly_page_ranks_that_months_days_only(built):
    out, _, _, _ = built
    html = (out / "monthly" / "2026-07" / "index.html").read_text(encoding="utf-8")
    body = text_of(html)
    assert "July 2026" in body
    # the ranking is over the month's own days, with the coverage of each model beside it
    assert "Ranked on this month" in body
    assert "Days" in body and "Coverage" in body
    # July has 31 days and the fixture scores every one of them
    assert "31" in body
    # no interval is quoted for a window that is not one of the published ones
    assert "no confidence interval is quoted here" in body
    # every model row links to its permanent page, which is where the intervals do live
    assert f'href="/station/{ALL_STATIONS}/model/exact/lead/1/"' in html


def test_a_monthly_page_states_completeness_qc_and_the_worst_day(built):
    out, _, _, _ = built
    html = (out / "monthly" / "2026-07" / "index.html").read_text(encoding="utf-8")
    body = text_of(html)
    assert "How complete the month was" in body
    assert "Instantaneous observations" in body and "Daily-extreme reports" in body
    # 2 stations x 31 days x 4 instants, all present in the fixture
    assert "248" in body and "100.0%" in body
    assert "Quality-control events" in body
    assert "Largest single-day error" in body
    assert re.search(r'href="/station/K[AB]{3}/model/\w+/lead/1/"', html)
    assert "Upstream changes" in body


def test_monthly_pages_are_announced_in_the_feed(built):
    out, _, _, _ = built
    root = ElementTree.fromstring((out / "feed.xml").read_text())
    ids = {c.text for e in root if e.tag.endswith("entry")
           for c in e if c.tag.endswith("id")}
    for month in MONTHS_IN_FIXTURE:
        assert f"tag:castcheck,{month}:monthly" in ids, month
    links = {c.get("href") for e in root if e.tag.endswith("entry")
             for c in e if c.tag.endswith("link")}
    assert f"{SITE_URL}/monthly/2026-07/" in links


def test_the_two_new_routes_are_in_the_navigation(built):
    out, _, _, _ = built
    for rel in ("index.html", "stations/index.html", "monthly/index.html",
                "diagnostics/index.html"):
        html = (out / rel).read_text(encoding="utf-8")
        assert 'href="/diagnostics/"' in html, rel
        assert 'href="/monthly/"' in html, rel
    assert 'aria-current="page"' in (out / "monthly" / "index.html").read_text()
    assert 'aria-current="page"' in (out / "diagnostics" / "index.html").read_text()


def test_the_new_pages_survive_a_build_with_no_data(tmp_path):
    """Neither route may 404 while the record is short, and neither may crash on an empty frame."""
    build_site(as_of="2026-08-30", out=tmp_path, scores=pd.DataFrame(),
               pairwise=pd.DataFrame(), daily=pd.DataFrame(columns=DAILY_COLUMNS),
               truth=pd.DataFrame(columns=TRUTH_COLUMNS), stations=STATION_OBJS,
               models=MODEL_OBJS, truth_instant=pd.DataFrame())
    diag = text_of((tmp_path / "diagnostics" / "index.html").read_text())
    assert "not yet attributed" in diag
    assert "No instant has enough scored days yet" in diag
    monthly = text_of((tmp_path / "monthly" / "index.html").read_text())
    assert "No calendar month has finished" in monthly
    assert not list((tmp_path / "monthly").glob("2*"))


def test_complete_months_never_include_the_month_in_progress():
    from datetime import date as _date

    from castcheck.site.build import _complete_months

    days = pd.Series(pd.date_range("2026-05-20", "2026-08-15", freq="D"))
    assert _complete_months(days, _date(2026, 8, 15)) == ["2026-05", "2026-06", "2026-07"]
    # the last day of a month is enough, one short is not
    assert _complete_months(days, _date(2026, 7, 31))[-1] == "2026-07"
    assert _complete_months(days, _date(2026, 7, 30))[-1] == "2026-06"
    assert _complete_months(pd.Series(dtype="datetime64[ns]"), _date(2026, 8, 15)) == []


# ------------------------------------------------------------------- citation

def test_the_long_citation_carries_the_concept_doi(built):
    """The concept DOI resolves to the newest version, so it is what a citation should name."""
    from castcheck.site.build import CONCEPT_DOI

    long = citation_long("KAAA", "Alpha Regional", "warm1", 1, "2026-08-29", "2026-08-29")
    assert f"doi:{CONCEPT_DOI}" in long
    # the short form stays one line: the permanent URL is the identifier there
    assert CONCEPT_DOI not in citation("KAAA", "warm1", 1, "2026-08-29")
    out, _, _, _ = built
    data = (out / "data" / "index.html").read_text(encoding="utf-8")
    assert f"doi:{CONCEPT_DOI}" in text_of(data)


# ------------------------------------------------------- final-review corrections (G3)

def test_the_star_says_which_family_it_is_the_lowest_of(built):
    """★ marks the leader among *ranked* models; below MIN_N a group is never in the family."""
    from castcheck.site.build import _MARK_TITLE

    assert "among ranked models" in _MARK_TITLE["★"]
    out, _, _, _ = built
    home = text_of((out / "index.html").read_text())
    assert f"lowest MAE among ranked models (n >= {MIN_N})" in home.replace("≥", ">=")
    feed = (out / "feed.xml").read_text()
    assert "lowest MAE among ranked models" in feed


def test_the_schema_table_documents_every_column_of_every_table(built):
    """A column with no entry printed an em dash where its meaning should be."""
    from castcheck.site.build import COLUMN_DOCS, COLUMN_DOCS_BY_TABLE
    from castcheck.store import DAILY_COLUMNS as DC

    for column in ("tmax_obs_s_c", "tmin_obs_s_c", "n_obs_samples", "native_overhang_h"):
        assert column in DC and COLUMN_DOCS[column][2], column
    # the same name means different things in different tables, so the docs are per table
    assert "mx2t3" in COLUMN_DOCS_BY_TABLE["forecast_values"]["variable"][2]
    assert "ALL" not in COLUMN_DOCS["station_id"][2]
    assert "ALL" in COLUMN_DOCS_BY_TABLE["scores"]["station_id"][2]
    assert "not a forecast valid time" in COLUMN_DOCS_BY_TABLE["truth_instant"]["valid_time"][2]
    assert "ASOS_IEM" in COLUMN_DOCS_BY_TABLE["truth_instant"]["source"][2]
    out, _, _, _ = built
    data = text_of((out / "data" / "index.html").read_text())
    assert "native extreme accumulation window reaches" in data
    assert f"one pre-built file per leaderboard view ({len(LEADERBOARD_VIEWS)} of them)" in data


def test_openapi_paths_resolve_against_a_server_that_serves_them():
    """cards.json is not under /api/v1; composed against the default server it 404s."""
    doc = openapi_document()
    cards = doc["paths"]["/station/{station}/cards.json"]["get"]
    assert cards["servers"] == [{"url": SITE_URL}]
    assert doc["servers"] == [{"url": f"{SITE_URL}/api/v1"}]
    view = doc["paths"]["/leaderboard/{view}.json"]["get"]["parameters"][0]["example"]
    assert view in {f"{w}-{i:02d}z-{m}-{v}" for w, i, m, v in LEADERBOARD_VIEWS}, \
        "the example must name a view that is still built"
    # status.json is a completeness report, not a score envelope
    status = doc["paths"]["/status.json"]["get"]
    ref = status["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/Status")
    props = doc["components"]["schemas"]["Status"]["properties"]
    for absent in ("window", "units", "method"):
        assert absent not in props, absent
    assert "n_current_gaps" in props and "gaps_today" in props


def test_pages_state_scope_without_overclaiming(built):
    out, _, _, _ = built
    stations = (out / "stations" / "index.html").read_text()
    assert "#11-how-the-23-stations-were-chosen" in stations
    body = text_of(stations)
    assert "New York Central Park" in body and "22 of the" in body
    assert "all 23 are major city airports" not in body
    # the persistence baseline is lagged, so "yesterday" is only true at lead day 1
    home = text_of((out / "index.html").read_text())
    assert "lead_day days earlier" in home
    assert "yesterday's observation" not in home
    # a green status bar next to a bare "Gaps 253" read as 253 open faults
    status = text_of((out / "status" / "index.html").read_text())
    assert "Historical gaps" in status and "open today" in status
    # every numeric column on the permanent link names its unit
    card = text_of((out / "station" / "KAAA" / "model" / "warm1" / "lead" / "1"
                    / "index.html").read_text())
    assert "MAE debiased °F" in card


def test_the_lead_day_limitation_matches_the_six_hour_spread():
    from castcheck.site.build import LIMITATIONS

    lead = next(x for x in LIMITATIONS if "same lead day covers different forecast hours" in x)
    assert "six hours" in lead and "three hours" not in lead


def test_only_a_dated_snapshot_becomes_a_feed_entry(tmp_path, monkeypatch):
    """A sync conflict copy once put an unparseable timestamp into every reader's feed."""
    import castcheck.site.build as build_mod

    hist = tmp_path / "data" / "scores" / "history"
    hist.mkdir(parents=True)
    for name in ("2026-08-29.parquet", "2026-08-30.parquet", "2026-08-30 2.parquet",
                 "latest.parquet"):
        (hist / name).write_bytes(b"")
    monkeypatch.setattr(build_mod, "REPO_ROOT", tmp_path)
    entries = build_mod._feed_entries(pd.DataFrame(columns=["station_id", "window", "init_hour",
                                                           "method", "variable", "lead_day",
                                                           "model_id", "n", "mae"]),
                                      {}, "2026-08-30", "2026-08-30T11:00:00+00:00")
    assert [e["date"] for e in entries] == ["2026-08-30", "2026-08-29"]
    for e in entries:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["date"])
