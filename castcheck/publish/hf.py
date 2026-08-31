"""Push the data/ shards to a Hugging Face dataset repo (DESIGN §7, METHODOLOGY §8).

Token: HF_TOKEN env or ~/.cache/huggingface/token. The dataset card is regenerated on every push.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from .. import METHODOLOGY_VERSION, SCHEMA_VERSION
from ..config import DATA_DIR, REPO_ROOT, load_models, load_stations

CARD = """---
license: cc-by-4.0
pretty_name: CastCheck — U.S. station-level temperature forecast verification
tags:
  - weather
  - forecast-verification
  - climate
  - time-series
  - ecmwf
  - gfs
  - graphcast
  - pangu-weather
  - aifs
size_categories:
  - 1M<n<10M
---

# CastCheck — daily station-level verification of public weather forecasts

Independent, automated verification of **raw** 2 m temperature forecasts from operational NWP
(ECMWF IFS HRES, NCEP GFS) and AI models (ECMWF AIFS Single; NOAA/CIRA operational runs of GraphCast,
Pangu-Weather, FourCastNet v2 and Aurora from both GFS and IFS initial conditions) at {n_stations}
U.S. airport stations, scored against the NWS Daily Climate Report (CLI).

- Site and permanent links: https://castcheck.zifanzhang.com
- Methodology v{mv}: https://castcheck.zifanzhang.com/methodology/
- Code: https://github.com/ZzZTripleZzZ/castcheck
- Updated daily by an automated pipeline. Last push: {ts}

## Fairness statement
These are raw model outputs at 0.25°, without MOS, bias correction, or downscaling. They are not
equivalent to the post-processed products end users receive, and scores here understate operational
forecast quality. Daily extremes are computed identically for every model from the common 6-hourly
instantaneous samples (see methodology §2.3).

## Files (schema v{sv})
- `data/forecast_values/model_id=*/year_month=*.parquet` — extracted station values (long format)
- `data/truth_daily/year=*.parquet` — NWS CLI/CF6/observation truth with first-final policy and QC flags
- `data/daily_forecasts/model_id=*/year=*.parquet` — derived sampled/native daily extremes per lead day
- `data/scores/latest.parquet`, `data/scores/pairwise_latest.parquet` — published aggregates with bootstrap CIs

```python
from datasets import load_dataset
ds = load_dataset("{repo}", data_files="data/scores/latest.parquet")
```

## Sources and licences
ECMWF Open Data (CC-BY-4.0) · NOAA/NCEP GFS (public domain) · NOAA/CIRA AIWP (open data; cite
the AIWP BAMS paper) · NWS climate reports (public domain) · Iowa Environmental Mesonet AFOS archive.
This dataset is released under CC-BY-4.0; please cite as
*CastCheck, methodology v{mv}, https://castcheck.zifanzhang.com (accessed YYYY-MM-DD)*.

## Models
{models}

## Stations
{stations}
"""


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


def dataset_card(repo: str) -> str:
    models = "\n".join(f"- `{m.model_id}` — {m.family}" + (f" ({m.init_field} initial conditions)" if m.init_field else "")
                       for m in load_models())
    stations = "\n".join(f"- `{s.id}` {s.name} ({s.lat:.3f}, {s.lon:.3f}, {s.elev_m} m)" for s in load_stations())
    return CARD.format(
        n_stations=len(load_stations()), mv=METHODOLOGY_VERSION, sv=SCHEMA_VERSION, repo=repo,
        ts=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), models=models, stations=stations,
    )


#: Uploaded to the dataset repo; `raw/` (logs, the run journal) and half-written shards stay local.
ALLOW_PATTERNS = ["**/*.parquet"]
IGNORE_PATTERNS = ["raw/**", "**/*.tmp.parquet", "**/.*"]


def files_to_push() -> list[Path]:
    """The shards :func:`push_dataset` would upload, in repo order (used by ``--dry-run``)."""
    keep: list[Path] = []
    for p in sorted(DATA_DIR.rglob("*.parquet")):
        rel = p.relative_to(DATA_DIR).as_posix()
        if rel.startswith("raw/") or p.name.startswith(".") or p.name.endswith(".tmp.parquet"):
            continue
        keep.append(p)
    return keep


def push_dataset(repo: str, private: bool = False, dry_run: bool = False) -> str:
    """Upload the dataset card, METHODOLOGY.md and every committed parquet shard.

    One commit per day is intended: the repo history is the publication log, and ~365 commits a year
    is well within what the Hub handles. `dry_run` reports the file list and the card without
    contacting the Hub at all, so it is safe to run against the public repo name.
    """
    tok = _token()
    if dry_run:
        files = files_to_push()
        total = sum(f.stat().st_size for f in files)
        card = dataset_card(repo)
        return (f"dry-run: would push {len(files)} parquet file(s), {total / 1e6:.1f} MB, plus "
                f"README.md ({len(card)} chars) and METHODOLOGY.md to {repo} "
                f"(private={private}, token={'present' if tok else 'MISSING'})")
    if not tok:
        return "skipped: no HF token"

    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    card_path = REPO_ROOT / ".hf_README.md"
    card_path.write_text(dataset_card(repo), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="update dataset card")
    api.upload_file(path_or_fileobj=str(REPO_ROOT / "METHODOLOGY.md"), path_in_repo="METHODOLOGY.md", repo_id=repo,
                    repo_type="dataset", commit_message="methodology")
    info = api.upload_folder(
        folder_path=str(DATA_DIR), path_in_repo="data", repo_id=repo, repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS, ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"data update {datetime.now(UTC):%Y-%m-%d}",
    )
    card_path.unlink(missing_ok=True)
    return f"pushed to https://huggingface.co/datasets/{repo} ({getattr(info, 'commit_url', info)})"
