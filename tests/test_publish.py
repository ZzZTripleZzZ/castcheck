"""Tests for the optional publishing targets (DESIGN §7).

Nothing here contacts a network service: the Bluesky chart is rendered from a synthetic `scores`
table, and the Hugging Face / Kaggle paths are exercised through their ``dry_run`` branches.
"""

from __future__ import annotations

import pandas as pd
import pytest

from castcheck.publish import bluesky
from castcheck.verify import MIN_N, PERSISTENCE_ID


def _score_row(model_id: str, mae_c: float, n: int, window: str = "90d", **over) -> dict:
    row = {
        "station_id": "ALL", "model_id": model_id, "init_hour": 0, "lead_day": 1, "variable": "tmax",
        "method": "bilinear", "window": window, "n": n, "mae": mae_c, "bias": 0.0, "rmse": mae_c * 1.2,
        "hit1f": 0.3, "hit2f": 0.6, "hit3f": 0.8, "skill_persistence": 0.2,
        "mae_ci_low": mae_c * 0.9, "mae_ci_high": mae_c * 1.1, "bias_ci_low": -0.2, "bias_ci_high": 0.2,
        "period_start": "2026-06-01", "period_end": "2026-08-29", "computed_at": "2026-08-30T11:00:00+00:00",
        "methodology_version": "0.1", "schema_version": "0.1",
    }
    row.update(over)
    return row


def _scores(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


LEADERBOARD = _scores([
    _score_row("ifs_hres", 1.5, 88),
    _score_row("graphcast_ifs", 1.7, 88),
    _score_row("gfs", 1.9, 88),
    _score_row("aurora_gfs", 2.4, 7),          # below MIN_N: greyed out, never "best"
    _score_row(PERSISTENCE_ID, 3.5, 88),
])


# --------------------------------------------------------------------------- selection


def test_selection_is_all_stations_lead1_00z_tmax_bilinear():
    noise = _scores([
        _score_row("ifs_hres", 0.1, 88, station_id="KNYC"),
        _score_row("ifs_hres", 0.2, 88, lead_day=3),
        _score_row("ifs_hres", 0.3, 88, init_hour=12),
        _score_row("ifs_hres", 0.4, 88, variable="tmin"),
        _score_row("ifs_hres", 0.5, 88, method="nearest"),
        *LEADERBOARD.to_dict("records"),
    ])
    sel, window = bluesky.select_leaderboard(noise)
    assert window == "90d"
    assert list(sel["model_id"]) == ["ifs_hres", "graphcast_ifs", "gfs", "aurora_gfs", PERSISTENCE_ID]


def test_selection_falls_back_to_a_wider_window_when_90d_is_too_thin():
    thin = _scores([
        _score_row("gfs", 1.9, 4, window="90d"),
        _score_row("gfs", 2.1, 40, window="30d"),
    ])
    sel, window = bluesky.select_leaderboard(thin)
    assert window == "30d"
    assert int(sel.iloc[0]["n"]) == 40


def test_selection_returns_none_when_nothing_reaches_min_n():
    thin = _scores([_score_row("gfs", 1.9, MIN_N - 1, window=w) for w in ("90d", "30d", "all")])
    assert bluesky.select_leaderboard(thin) is None
    assert bluesky.select_leaderboard(pd.DataFrame()) is None


def test_post_daily_skips_without_data(monkeypatch):
    monkeypatch.setattr(bluesky, "load_scores", pd.DataFrame)
    assert bluesky.post_daily(dry_run=True) == "skipped: not enough data"


# --------------------------------------------------------------------------- post text


def test_post_text_leads_with_the_result_and_fits_bluesky():
    sel, window = bluesky.select_leaderboard(LEADERBOARD)
    text = bluesky.post_text(sel, window)
    assert len(text) <= bluesky.MAX_POST_CHARS
    first = text.splitlines()[0]
    assert first.startswith("Best raw Tmax forecast at lead day 1 over the last 90 days:")
    assert "ECMWF IFS HRES" in first        # display name, not the model_id
    assert "2.7 °F MAE" in first            # 1.5 °C -> 2.7 °F
    assert "n=88" in first
    assert text.splitlines()[-1] == bluesky.SITE
    assert "no post-processing" in text


def test_post_text_never_announces_the_persistence_baseline_as_best():
    sel, window = bluesky.select_leaderboard(_scores([
        _score_row(PERSISTENCE_ID, 1.0, 88),
        _score_row("gfs", 1.9, 88),
    ]))
    assert "NCEP GFS" in bluesky.post_text(sel, window).splitlines()[0]


def test_post_text_uses_the_init_field_in_the_display_name():
    sel, window = bluesky.select_leaderboard(_scores([_score_row("graphcast_ifs", 1.2, 88)]))
    assert "GraphCast (IFS init)" in bluesky.post_text(sel, window)


# --------------------------------------------------------------------------- chart


def test_chart_is_a_1200x675_png_with_a_descriptive_alt_text():
    pytest.importorskip("matplotlib")
    sel, window = bluesky.select_leaderboard(LEADERBOARD)
    png, alt = bluesky.build_chart(sel, window)

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    assert (width, height) == (1200, 675)

    assert "bar chart" in alt.lower()
    assert "ECMWF IFS HRES 2.7 (n=88)" in alt
    assert "confidence interval" in alt
    assert "data through 2026-08-29" in alt


def test_dry_run_writes_the_preview_outside_the_committed_data_tables(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    preview = tmp_path / "raw" / "bluesky_preview.png"
    monkeypatch.setattr(bluesky, "PREVIEW_PATH", preview)
    monkeypatch.setattr(bluesky, "load_scores", lambda: LEADERBOARD)
    monkeypatch.delenv("BSKY_HANDLE", raising=False)
    monkeypatch.delenv("BSKY_APP_PASSWORD", raising=False)

    out = bluesky.post_daily(dry_run=True)

    assert preview.exists() and preview.read_bytes()[:4] == b"\x89PNG"
    assert "raw" in preview.parts and "scores" not in preview.parts
    assert "--- text ---" in out and "--- alt ---" in out


# --------------------------------------------------------------------------- hf / kaggle


def test_hf_dry_run_reports_the_file_list_without_a_token(monkeypatch):
    from castcheck.publish import hf

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hf, "_token", lambda: None)
    out = hf.push_dataset("castcheck/_ci-test", dry_run=True)
    assert out.startswith("dry-run: would push")
    assert "token=MISSING" in out


def test_hf_never_uploads_the_raw_directory(tmp_path, monkeypatch):
    from castcheck.publish import hf

    (tmp_path / "raw").mkdir()
    (tmp_path / "scores").mkdir()
    (tmp_path / "raw" / "debug.parquet").write_bytes(b"x")
    (tmp_path / "scores" / "latest.parquet").write_bytes(b"x")
    (tmp_path / "scores" / "latest.tmp.parquet").write_bytes(b"x")
    monkeypatch.setattr(hf, "DATA_DIR", tmp_path)

    names = [p.name for p in hf.files_to_push()]
    assert names == ["latest.parquet"]


def test_kaggle_status_output_decides_create_vs_version(monkeypatch):
    import subprocess

    from castcheck.publish import kaggle

    class _R:
        def __init__(self, rc, out):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    cases = [(_R(0, "ready"), True), (_R(1, "404 - Not Found"), False),
             (_R(0, "404 - Not Found"), False), (_R(0, "pending"), True)]
    for result, expected in cases:
        monkeypatch.setattr(subprocess, "run", lambda *a, _r=result, **k: _r)
        assert kaggle.dataset_exists("slug", {}) is expected
