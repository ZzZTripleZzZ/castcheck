"""Mirror the published tables to a Kaggle dataset (entry point only; HF is the primary home).

Auth: KAGGLE_API_TOKEN env (new-style KGAT token) or ~/.kaggle/access_token.
Kaggle datasets accept a folder + dataset-metadata.json; we upload the *scores* and *daily* tables
as CSV/Parquet so that notebooks can start without the raw values.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .. import METHODOLOGY_VERSION
from ..config import DATA_DIR, REPO_ROOT

KAGGLE_USER = "zhangzifan716"


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

    src = DATA_DIR / "scores" / "latest.parquet"
    if src.exists():
        pd.read_parquet(src).to_csv(stage / "scores_latest.csv", index=False)
    src = DATA_DIR / "scores" / "pairwise_latest.parquet"
    if src.exists():
        shutil.copy(src, stage / "scores_pairwise_latest.parquet")  # ~50 MB as CSV; keep parquet
    daily_dir = DATA_DIR / "daily_forecasts"
    if daily_dir.exists():
        frames = [pd.read_parquet(f) for f in daily_dir.rglob("*.parquet") if ".tmp" not in f.name and not f.name.startswith(".")]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(stage / "daily_forecasts.parquet", index=False)
    truth_dir = DATA_DIR / "truth_daily"
    if truth_dir.exists():
        frames = [pd.read_parquet(f) for f in truth_dir.glob("*.parquet") if ".tmp" not in f.name and not f.name.startswith(".")]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(stage / "truth_daily.csv", index=False)
    shutil.copy(REPO_ROOT / "METHODOLOGY.md", stage / "METHODOLOGY.md")
    meta = {
        "title": "CastCheck: US temperature forecast verification",
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "subtitle": "Daily station-level errors of NWP and AI weather models vs NWS reports",
        "description": (
            "Independent daily verification of raw 2 m temperature forecasts at 23 U.S. airport stations. "
            f"Methodology v{METHODOLOGY_VERSION}: https://castcheck.zifanzhang.com/methodology/ . "
            "Primary home and full raw values: https://huggingface.co/datasets/castcheck/temperature-verification"
        ),
        "keywords": ["weather and climate", "earth and nature"],
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return stage


def _kaggle_bin() -> str:
    local = REPO_ROOT / ".venv" / "bin" / "kaggle"
    return str(local) if local.exists() else "kaggle"


def dataset_exists(slug: str, env: dict) -> bool:
    """Whether ``{KAGGLE_USER}/{slug}`` already exists, from ``kaggle datasets status``.

    The command prints the *status word* on stdout (``ready``, ``pending``, ``error``) for a dataset
    that exists, and ``404 - Not Found`` (exit code 1 on newer clients, 0 on older ones) for one
    that does not — so both the exit code and the text have to be checked. Anything unrecognised is
    treated as "exists" because ``create`` on an existing slug is a hard error, while ``version`` on
    a missing one merely fails with a clear message.
    """
    r = subprocess.run([_kaggle_bin(), "datasets", "status", f"{KAGGLE_USER}/{slug}"],
                       env=env, capture_output=True, text=True, check=False)
    text = (r.stdout + r.stderr).strip().lower()
    if "404" in text or "not found" in text or "does not exist" in text:
        return False
    return r.returncode == 0


def push_dataset(slug: str, dry_run: bool = False) -> str:
    env = _env()
    if "KAGGLE_API_TOKEN" not in env and not dry_run:
        return "skipped: no Kaggle token"
    stage = _stage(slug)
    files = sorted(p.name for p in stage.iterdir())
    if dry_run:
        size = sum(p.stat().st_size for p in stage.iterdir())
        shutil.rmtree(stage, ignore_errors=True)
        return (f"dry-run: staged {len(files)} file(s), {size / 1e6:.1f} MB for {KAGGLE_USER}/{slug} "
                f"({', '.join(files)}); token={'present' if 'KAGGLE_API_TOKEN' in env else 'MISSING'}")
    if dataset_exists(slug, env):
        cmd = [_kaggle_bin(), "datasets", "version", "-p", str(stage),
               "-m", f"update {datetime.now(UTC):%Y-%m-%d}", "-r", "zip"]
    else:
        cmd = [_kaggle_bin(), "datasets", "create", "-p", str(stage), "-r", "zip", "--public"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    shutil.rmtree(stage, ignore_errors=True)
    lines = [ln for ln in (r.stdout + "\n" + r.stderr).splitlines() if ln.strip() and "%|" not in ln]
    out = "\n".join(lines[-6:])
    if r.returncode != 0 or re.search(r"(creation|upload|version) error|^error|invalid|forbidden|unauthorized", out, re.I | re.M):
        raise RuntimeError(f"kaggle publish failed: {out}")
    return out
