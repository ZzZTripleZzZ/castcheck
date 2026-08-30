"""Mirror the published tables to a Kaggle dataset (entry point only; HF is the primary home).

Auth: KAGGLE_API_TOKEN env (new-style KGAT token) or ~/.kaggle/access_token.
Kaggle datasets accept a folder + dataset-metadata.json; we upload the *scores* and *daily* tables
as CSV/Parquet so that notebooks can start without the raw values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .. import METHODOLOGY_VERSION
from ..config import DATA_DIR, REPO_ROOT

KAGGLE_USER = "triplez716"


def _env() -> dict:
    env = dict(os.environ)
    if "KAGGLE_API_TOKEN" not in env:
        p = Path.home() / ".kaggle" / "access_token"
        if p.exists():
            env["KAGGLE_API_TOKEN"] = p.read_text().strip()
    return env


def _stage(slug: str) -> Path:
    stage = REPO_ROOT / ".kaggle_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    import pandas as pd

    for name in ("latest", "pairwise_latest"):
        src = DATA_DIR / "scores" / f"{name}.parquet"
        if src.exists():
            pd.read_parquet(src).to_csv(stage / f"scores_{name}.csv", index=False)
    daily_dir = DATA_DIR / "daily_forecasts"
    if daily_dir.exists():
        frames = [pd.read_parquet(f) for f in daily_dir.rglob("*.parquet")]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(stage / "daily_forecasts.parquet", index=False)
    truth_dir = DATA_DIR / "truth_daily"
    if truth_dir.exists():
        frames = [pd.read_parquet(f) for f in truth_dir.glob("*.parquet")]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(stage / "truth_daily.csv", index=False)
    shutil.copy(REPO_ROOT / "METHODOLOGY.md", stage / "METHODOLOGY.md")
    meta = {
        "title": "CastCheck: U.S. temperature forecast verification (NWP + AI models)",
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "subtitle": "Daily station-level errors of IFS, AIFS, GFS, GraphCast, Pangu, FourCastNet, Aurora vs NWS climate reports",
        "description": (
            "Independent daily verification of raw 2 m temperature forecasts at 23 U.S. airport stations. "
            f"Methodology v{METHODOLOGY_VERSION}: https://castcheck.zifanzhang.com/methodology/ . "
            "Primary home and full raw values: https://huggingface.co/datasets/castcheck/temperature-verification"
        ),
        "keywords": ["weather", "forecast verification", "time series", "climate", "ecmwf", "gfs", "ai weather models"],
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return stage


def push_dataset(slug: str) -> str:
    env = _env()
    if "KAGGLE_API_TOKEN" not in env:
        return "skipped: no Kaggle token"
    stage = _stage(slug)
    kaggle = str(REPO_ROOT / ".venv" / "bin" / "kaggle") if (REPO_ROOT / ".venv" / "bin" / "kaggle").exists() else "kaggle"
    exists = subprocess.run([kaggle, "datasets", "status", f"{KAGGLE_USER}/{slug}"], env=env, capture_output=True, text=True, check=False)
    if exists.returncode == 0 and "ready" in (exists.stdout + exists.stderr).lower():
        cmd = [kaggle, "datasets", "version", "-p", str(stage), "-m", f"update {datetime.now(UTC):%Y-%m-%d}", "-r", "zip"]
    else:
        cmd = [kaggle, "datasets", "create", "-p", str(stage), "-r", "zip", "--public"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    shutil.rmtree(stage, ignore_errors=True)
    return (r.stdout + r.stderr).strip()[-500:]
