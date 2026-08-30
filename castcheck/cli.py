"""CastCheck command line (DESIGN §5). All commands are idempotent.

Imports of heavy modules are done inside commands so that a missing optional dependency (or a module
still under construction) never breaks unrelated commands.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta

import typer

from . import __version__
from .config import PUBLIC_DIR, load_models, load_stations, model_by_id

app = typer.Typer(add_completion=False, no_args_is_help=True, help="CastCheck — forecast verification pipeline")
publish_app = typer.Typer(no_args_is_help=True, help="Optional publishing targets (token-gated)")
app.add_typer(publish_app, name="publish")

# Expected availability delay (hours after init) before a run is worth fetching; measured 2026-08-30.
AVAILABILITY_DELAY_H = {"ecmwf": 8.0, "gfs": 5.5, "aiwp": 9.0}


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _parse_init(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", ""))
    return dt.replace(tzinfo=UTC)


def _source_for(model):
    if model.source == "ecmwf":
        import os

        from .sources.ecmwf import EcmwfSource

        # S3 answers bursts of range GETs with 503 SlowDown; 4 concurrent requests is measured safe.
        return EcmwfSource(workers=int(os.environ.get("CASTCHECK_ECMWF_WORKERS", "4")))
    if model.source == "gfs":
        from .sources.gfs import GfsSource

        return GfsSource()
    if model.source == "aiwp":
        from .sources.aiwp import AiwpSource

        return AiwpSource()
    raise typer.BadParameter(f"unknown source {model.source}")


def _fetch_one(model, init: datetime, station_ids: list[str] | None) -> tuple[str, int, int]:
    from .sources.base import FetchRequest
    from .store import upsert_forecast_values

    stations = load_stations()
    if station_ids:
        stations = [s for s in stations if s.id in station_ids]
    res = _source_for(model).fetch_run(FetchRequest(model=model, init_time=init, stations=stations))
    upsert_forecast_values(res.rows)
    n_missing = int((res.rows["missing_reason"] != "").sum()) if len(res.rows) else 0
    return model.model_id, res.n_present, n_missing


@app.callback()
def _root():
    """CastCheck v{__version__}"""


@app.command()
def version():
    typer.echo(__version__)


@app.command()
def fetch(model: str = typer.Option(..., help="model_id"), init: str = typer.Option(..., help="e.g. 2026-08-30T00"),
          stations: str = typer.Option("", help="comma-separated ICAO subset")):
    """Fetch one model run and upsert station values."""
    m = model_by_id(model)
    ids = [s for s in stations.split(",") if s] or None
    mid, present, missing = _fetch_one(m, _parse_init(init), ids)
    typer.echo(f"{mid} {init}: present={present} missing={missing}")


@app.command("fetch-latest")
def fetch_latest(lookback_days: int = 3, workers: int = 3, models: str = typer.Option("", help="comma-separated model_ids")):
    """Fetch every run that should be available by now (last `lookback_days`) and is not stored yet."""
    from .store import existing_inits

    now = _now()
    wanted = [m for m in load_models() if not models or m.model_id in models.split(",")]
    jobs: list[tuple] = []
    for m in wanted:
        have = existing_inits(m.model_id)
        for d in range(lookback_days + 1):
            day = (now - timedelta(days=d)).date()
            for hh in m.inits:
                init = datetime(day.year, day.month, day.day, hh, tzinfo=UTC)
                if init + timedelta(hours=AVAILABILITY_DELAY_H.get(m.source, 8.0)) > now:
                    continue
                if any(abs((h.to_pydatetime() - init).total_seconds()) < 1 for h in have):
                    continue
                jobs.append((m, init))
    if not jobs:
        typer.echo("nothing to fetch")
        return
    typer.echo(f"{len(jobs)} run(s) to fetch")
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, m, init, None): (m.model_id, init) for m, init in jobs}
        for f in as_completed(futs):
            mid, init = futs[f]
            try:
                _, present, missing = f.result()
                typer.echo(f"  {mid} {init:%Y-%m-%dT%H}Z present={present} missing={missing}")
                if present == 0:
                    failures += 1
            except Exception as e:  # noqa: BLE001 — never let one run kill the batch
                failures += 1
                typer.echo(f"  {mid} {init:%Y-%m-%dT%H}Z ERROR {type(e).__name__}: {e}", err=True)
    if failures:
        typer.echo(f"{failures} run(s) had no data (will be retried by later runs / backfill)", err=True)


@app.command()
def backfill(model: str, start: str, end: str, workers: int = 2, skip_existing: bool = True):
    """Fetch all inits of a model between two dates (inclusive), e.g. --start 2025-01-01 --end 2025-01-31."""
    from .store import existing_inits

    m = model_by_id(model)
    have = existing_inits(m.model_id) if skip_existing else set()
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    inits = []
    d = d0
    while d <= d1:
        for hh in m.inits:
            init = datetime(d.year, d.month, d.day, hh, tzinfo=UTC)
            if not any(abs((h.to_pydatetime() - init).total_seconds()) < 1 for h in have):
                inits.append(init)
        d += timedelta(days=1)
    typer.echo(f"{m.model_id}: {len(inits)} run(s) to backfill")
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, m, init, None): init for init in inits}
        for f in as_completed(futs):
            init = futs[f]
            try:
                _, present, missing = f.result()
                ok += present > 0
                typer.echo(f"  {init:%Y-%m-%dT%H}Z present={present} missing={missing}")
            except Exception as e:  # noqa: BLE001
                typer.echo(f"  {init:%Y-%m-%dT%H}Z ERROR {e}", err=True)
    typer.echo(f"done: {ok}/{len(inits)} runs with data")


@app.command()
def truth(day: str = typer.Option("", "--date", help="climatological date YYYY-MM-DD (default: UTC yesterday)")):
    """Fetch the final NWS CLI report (with CF6/obs fallback and QC) for all stations for one day."""
    from .store import upsert_truth
    from .truth import truth_for_date

    climo = date.fromisoformat(day) if day else (_now().date() - timedelta(days=1))
    rows = truth_for_date(load_stations(), climo)
    out = upsert_truth(rows)
    n_cli = int(((rows["source"] == "CLI") & rows["is_final"]).sum()) if len(rows) else 0
    typer.echo(f"{climo}: {len(rows)} rows ({n_cli} final CLI) -> {out}")


@app.command("truth-backfill")
def truth_backfill(start: str, end: str, stations: str = ""):
    from .store import upsert_truth
    from .truth import truth_backfill as _tb

    sts = load_stations()
    if stations:
        sts = [s for s in sts if s.id in stations.split(",")]
    rows = _tb(sts, date.fromisoformat(start), date.fromisoformat(end))
    typer.echo(f"{len(rows)} rows -> {upsert_truth(rows)}")


@app.command()
def derive():
    """Recompute daily_forecasts from forecast_values (full re-derivation, idempotent)."""
    from .derive import daily_from_values
    from .store import read_forecast_values, write_daily

    values = read_forecast_values()
    daily = daily_from_values(values, load_stations(), load_models())
    typer.echo(f"{len(values)} values -> {len(daily)} daily rows -> {write_daily(daily)}")


@app.command()
def verify(n_boot: int = 1000):
    """Compute scores and pairwise comparisons from daily_forecasts + truth."""
    from .store import read_daily, read_truth, write_scores
    from .verify import persistence_daily, score

    daily = read_daily()
    tr = read_truth()
    import pandas as pd

    daily = pd.concat([daily, persistence_daily(tr)], ignore_index=True) if len(tr) else daily
    scores, pairwise = score(daily, tr, n_boot=n_boot)
    as_of = _now().strftime("%Y-%m-%d")
    write_scores(scores, pairwise, as_of)
    typer.echo(f"scores={len(scores)} pairwise={len(pairwise)} as_of={as_of}")


@app.command("build-site")
def build_site():
    """Generate public/ (HTML + JSON API + status)."""
    from .api import export_api
    from .site.build import build_site as _build
    from .status import write_status
    from .store import read_scores

    scores, pairwise = read_scores()
    export_api(scores, pairwise, load_stations(), load_models())
    write_status(_now())
    _build(_now())
    typer.echo(f"site written to {PUBLIC_DIR}")


@app.command()
def status(fail_on_gaps: bool = True):
    """Write public/api/v1/status.json; exit 1 if today's runs or truth are incomplete."""
    from .status import write_status

    rep = write_status(_now())
    gaps = rep.get("gaps_today", [])
    typer.echo(f"gaps today: {len(gaps)}")
    for g in gaps[:50]:
        typer.echo(f"  {g}")
    if gaps and fail_on_gaps:
        sys.exit(1)


@app.command()
def daily(n_boot: int = 1000):
    """derive -> verify -> build-site in one go (used by the daily workflow)."""
    derive()
    verify(n_boot=n_boot)
    build_site()


@publish_app.command("hf")
def publish_hf(repo: str = "castcheck/temperature-verification", private: bool = False):
    from .publish.hf import push_dataset

    typer.echo(push_dataset(repo, private=private))


@publish_app.command("kaggle")
def publish_kaggle(slug: str = "castcheck-temperature-forecast-verification"):
    from .publish.kaggle import push_dataset

    typer.echo(push_dataset(slug))


@publish_app.command("bluesky")
def publish_bluesky(dry_run: bool = False):
    from .publish.bluesky import post_daily

    typer.echo(post_daily(dry_run=dry_run))


if __name__ == "__main__":
    app()
