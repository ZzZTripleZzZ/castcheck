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
U.S. first-order stations — 22 major airports plus New York Central Park. The headline metric is the instantaneous 2 m temperature at 00/06/12/18 UTC
against the ASOS observation at the same instants; daily extremes are published as secondary
variables, both like-for-like and against the NWS Daily Climate Report (CLI).

- Site and permanent links: https://castcheck.zifanzhang.com
- Methodology v{mv}: https://castcheck.zifanzhang.com/methodology/
- Code: https://github.com/ZzZTripleZzZ/castcheck
- Archived: https://doi.org/10.5281/zenodo.22212363
- Updated daily by an automated pipeline. Last push: {ts}

## Fairness statement
These are raw model outputs at 0.25°, without MOS, bias correction, or downscaling. They are not
equivalent to the post-processed products end users receive, and scores here understate operational
forecast quality. Every model is sampled at the same four instants, and the headline metric compares
those samples with observations at the same instants, so no part of it depends on a model's own
diurnal amplitude. The extremes verified against the CLI report (`tmax_cli`, `tmin_cli`) carry a
sampling penalty whose size does differ between models, so they are secondary and are never used to
rank (see methodology §2.3).

## Files (schema v{sv})
- `data/forecast_values/model_id=*/year_month=*.parquet` — extracted station values (long format)
- `data/truth_instant/year=*.parquet` — observed 2 m temperature at 00/06/12/18 UTC, with QC flags
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
ECMWF data is © ECMWF, licensed under CC-BY-4.0 and used unmodified apart from interpolation to the
station; ECMWF does not endorse this work. This dataset is released under CC-BY-4.0; please cite as
*CastCheck, methodology v{mv}, doi:10.5281/zenodo.22212363,
https://castcheck.zifanzhang.com (accessed YYYY-MM-DD)*.

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
# `upload_folder` never removes remote files, so a macOS/iCloud conflict copy ("name 2.parquet") that
# was pushed once stays on the Hub forever and silently duplicates a shard for anyone globbing the
# tree. Prune exactly those names — and nothing else, so a partial local data/ can never delete a
# legitimate shard from the published dataset.
DELETE_PATTERNS = ["**/* [0-9].parquet", "**/* [0-9][0-9].parquet"]


def files_to_push() -> list[Path]:
    """The shards :func:`push_dataset` would upload, in repo order (used by ``--dry-run``)."""
    keep: list[Path] = []
    for p in sorted(DATA_DIR.rglob("*.parquet")):
        rel = p.relative_to(DATA_DIR).as_posix()
        if rel.startswith("raw/") or p.name.startswith(".") or p.name.endswith(".tmp.parquet"):
            continue
        keep.append(p)
    return keep


def squash_is_due(now: datetime, force: str | None = None) -> bool:
    """Whether :func:`maybe_squash` should consider squashing at ``now``.

    The window is 1 January (UTC) only. ``CASTCHECK_HF_SQUASH`` overrides it: ``1`` forces the
    attempt on any date (used to run the annual squash by hand after a botched year), ``0`` disables
    it entirely (used by anyone mirroring to a repo whose history they want kept).
    """
    flag = force if force is not None else os.environ.get("CASTCHECK_HF_SQUASH", "")
    if flag == "1":
        return True
    if flag == "0":
        return False
    return now.month == 1 and now.day == 1


def maybe_squash(api, repo: str, now: datetime | None = None) -> str:
    """Once a year, collapse the dataset repo's git history into a single commit.

    **Why.** One data commit a day is ~365 commits and — because every push rewrites the same
    parquet shards — roughly a year of *superseded* shard blobs a year. The Hub keeps every blob
    forever, so an un-squashed decade would make ``git clone`` of the dataset download ten years of
    dead revisions to reconstruct one day's table. ``super_squash_history`` throws the dead blobs
    away and leaves the working tree byte-for-byte identical.

    **Risks — read before changing the schedule.**

    * It is **irreversible and non-fast-forward**: every past commit SHA disappears. Anyone who
      pinned ``revision="<sha>"`` or ``revision="refs/convert/parquet"`` against an old commit gets
      a 404 afterwards. That is the deliberate trade: CastCheck's citable snapshots are the dated
      files inside ``data/scores/history/`` and the tagged releases in the *code* repo, never Hub
      commit SHAs — see DESIGN §7 and the dataset card.
    * It cannot be undone by us, only by the Hub's support.
    * It needs a write token with enough scope; on failure this returns a note and the caller
      continues to the normal upload, so a squash that is refused never blocks the day's data.
    * It is run **before** the day's upload, so the first commit of the new year is the one that
      re-establishes history. Running it after would leave a squash commit as the tip and make the
      day's data a second commit — same result, one more round trip.

    Idempotent by construction: after a successful squash the repo has exactly one commit, so the
    ``> 1`` test below is false for every later push on the same day.
    """
    now = now or datetime.now(UTC)
    if not squash_is_due(now):
        return ""
    try:
        commits = api.list_repo_commits(repo_id=repo, repo_type="dataset")
        n = len(list(commits))
    except Exception as exc:  # noqa: BLE001 - never let housekeeping break the daily push
        return f"squash skipped: could not list commits ({exc})"
    if n <= 1:
        return f"squash not needed: {repo} already has {n} commit(s)"
    try:
        api.super_squash_history(
            repo_id=repo, repo_type="dataset",
            commit_message=f"annual history squash {now:%Y-%m-%d} (CastCheck; see DESIGN §7)",
        )
    except Exception as exc:  # noqa: BLE001
        return f"squash FAILED (continuing with the upload): {exc}"
    return f"squashed {n} commit(s) of {repo} into one (annual, {now:%Y-%m-%d}); history is gone"


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
    squash = maybe_squash(api, repo)
    if squash:
        print(squash)  # noqa: T201 - the confirmation line the workflow log must carry
    card_path = REPO_ROOT / ".hf_README.md"
    card_path.write_text(dataset_card(repo), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="update dataset card")
    api.upload_file(path_or_fileobj=str(REPO_ROOT / "METHODOLOGY.md"), path_in_repo="METHODOLOGY.md", repo_id=repo,
                    repo_type="dataset", commit_message="methodology")
    info = api.upload_folder(
        folder_path=str(DATA_DIR), path_in_repo="data", repo_id=repo, repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS, ignore_patterns=IGNORE_PATTERNS, delete_patterns=DELETE_PATTERNS,
        commit_message=f"data update {datetime.now(UTC):%Y-%m-%d}",
    )
    card_path.unlink(missing_ok=True)
    out = f"pushed to https://huggingface.co/datasets/{repo} ({getattr(info, 'commit_url', info)})"
    return f"{out}\n{squash}" if squash else out
