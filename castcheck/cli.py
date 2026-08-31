"""CastCheck command line (DESIGN §5). All commands are idempotent.

Imports of heavy modules are done inside commands so that a missing optional dependency (or a module
still under construction) never breaks unrelated commands.

Every command configures logging once (INFO to stderr, ``-v`` for DEBUG, ``CASTCHECK_LOG_JSON=1``
for one JSON object per line), prints a single summary line when it finishes, and appends its
outcome to ``data/raw/last_run.json`` (DESIGN §7.1) so the status page can show how fresh each part
of the pipeline is.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta

import typer

from . import __version__
from .config import PUBLIC_DIR, REPO_ROOT, ModelSpec, load_models, load_stations, model_by_id

app = typer.Typer(add_completion=False, no_args_is_help=True, help="CastCheck — forecast verification pipeline")
publish_app = typer.Typer(no_args_is_help=True, help="Optional publishing targets (token-gated)")
app.add_typer(publish_app, name="publish")

log = logging.getLogger("castcheck")

# Expected availability delay (hours after init) before a run is worth fetching; measured 2026-08-30.
# Keyed by ``source``; the same delay applies to the 00Z and 12Z cycles of a model.
AVAILABILITY_DELAY_H = {"ecmwf": 8.0, "gfs": 5.5, "aiwp": 9.5}
# AIWP publishes the GFS-initialised runs about three hours before the IFS-initialised ones, because
# it has to wait for ECMWF's own dissemination first.
AIWP_DELAY_H_BY_INIT_FIELD = {"GFS": 6.0, "IFS": 9.5}

#: A run that is stored but incomplete is retried by later passes; not more often than this, so that
#: four scheduled fetches a day do not re-download the same permanently-partial run four times.
DEFAULT_MIN_RETRY_H = 3.0


# --------------------------------------------------------------------------- logging / journal


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shipping (``CASTCHECK_LOG_JSON=1``)."""

    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger: INFO (or DEBUG) to **stderr**, so stdout stays machine-readable."""
    level = logging.DEBUG if verbose or os.environ.get("CASTCHECK_DEBUG") else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("CASTCHECK_LOG_JSON") == "1":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                               datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


@contextmanager
def _journal(command: str):
    """Time a command, print/record a one-line summary, and record failures too.

    The body appends to the yielded list; whatever it contains becomes the summary line.
    """
    from .store import record_run

    parts: list[str] = []
    started = datetime.now(UTC)
    t0 = time.monotonic()
    status, code = "ok", 0
    try:
        yield parts
    except typer.Exit as exc:  # a command that chose its own exit code (e.g. `status`)
        code = int(getattr(exc, "exit_code", 0) or 0)
        status = "ok" if code == 0 else "error"
        raise
    except SystemExit as exc:
        code = int(exc.code or 0)
        status = "ok" if code == 0 else "error"
        raise
    except Exception as exc:
        status, code = "error", 1
        detail = redact(exc, limit=160)
        parts.append(f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__)
        raise
    finally:
        dt = time.monotonic() - t0
        summary = redact("; ".join(p for p in parts if p), limit=500)
        log.info("%s %s in %.1fs — %s", command, status, dt, summary or "(no summary)")
        record_run(command, status=status, summary=summary, started_at=started, duration_s=dt,
                   exit_code=code)
        try:
            from .sources import _http

            _http.log_summary()
        except ImportError:  # pragma: no cover - _http is always importable in practice
            pass


# --------------------------------------------------------------------------- redaction

#: ``data/raw/last_run.json`` is committed to a public repository, so everything that reaches it is
#: scrubbed first. These patterns are ordered: query strings go before token shapes, so that a
#: credential carried as a URL parameter is removed with the query rather than left behind as a
#: bare word.
_REDACTIONS = (
    # ``https://host/path?key=secret`` -> ``https://host/path?…``
    (re.compile(r"(https?://[^\s?#]+)\?[^\s]*"), r"\1?…"),
    # long opaque strings: API keys, bearer tokens, AWS ids, signatures
    (re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{24,}\b"), "…"),
    # anything that names itself a secret, whatever it holds
    (re.compile(r"(?i)\b(token|secret|password|passwd|api[_-]?key|authorization)\b\s*[:=]\s*\S+"),
     r"\1=…"),
)


def redact(text: str, limit: int = 300) -> str:
    """One short, safe line: no absolute paths, no query strings, no token-shaped words.

    Journal entries and their summaries end up in a public file, and the things that land in them
    are exception messages — the least curated strings in the process. An upstream library is free
    to put a signed URL, a home directory or an API key into ``str(exc)``, so the journal treats
    every message as untrusted rather than trusting each library not to.
    """
    s = " ".join(str(text).split())
    if not s:
        return ""
    # absolute paths inside the repo become repo-relative; any other absolute path keeps its last
    # two components only, which is enough to identify the file and says nothing about the machine
    s = s.replace(str(REPO_ROOT) + os.sep, "").replace(str(REPO_ROOT), ".")
    s = re.sub(r"(?:/[^/\s:,'\"]+){3,}", lambda m: ".../" + "/".join(m.group(0).split("/")[-2:]), s)
    for pattern, repl in _REDACTIONS:
        s = pattern.sub(repl, s)
    return s[:limit]


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _parse_init(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", ""))
    return dt.replace(tzinfo=UTC)


def availability_delay_h(model: ModelSpec) -> float:
    """Hours after initialisation at which `model`'s run is normally complete upstream."""
    if model.source == "aiwp":
        return AIWP_DELAY_H_BY_INIT_FIELD.get(model.init_field or "GFS", AVAILABILITY_DELAY_H["aiwp"])
    return AVAILABILITY_DELAY_H.get(model.source, 8.0)


def _source_for(model: ModelSpec):
    if model.source == "ecmwf":
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


def _fetch_one(model: ModelSpec, init: datetime, station_ids: list[str] | None) -> tuple[str, int, int]:
    from .sources.base import FetchRequest
    from .store import upsert_forecast_values

    stations = load_stations()
    if station_ids:
        stations = [s for s in stations if s.id in station_ids]
    res = _source_for(model).fetch_run(FetchRequest(model=model, init_time=init, stations=stations))
    upsert_forecast_values(res.rows)
    n_missing = int((res.rows["missing_reason"] != "").sum()) if len(res.rows) else 0
    for note in res.notes:
        log.debug("%s %s: %s", model.model_id, init.isoformat(), note)
    return model.model_id, res.n_present, n_missing


def plan_runs(models: list[ModelSpec], now: datetime, lookback_days: int,
              have: Callable[[ModelSpec, str, str], set], last_attempt: Callable[[ModelSpec, str, str], dict],
              min_retry_h: float = DEFAULT_MIN_RETRY_H) -> list[tuple[ModelSpec, datetime]]:
    """The (model, init) runs `fetch-latest` should fetch now. Pure, so it is testable without I/O.

    A run is planned when (a) enough time has passed since its initialisation that upstream should
    have published it, (b) it is not already stored complete, and (c) it was not already attempted
    within the last `min_retry_h` hours — otherwise a run that upstream never completes would be
    re-downloaded by every scheduled pass until it drops out of the lookback window.
    """
    start = (now - timedelta(days=lookback_days)).date().isoformat()
    end = now.date().isoformat()
    jobs: list[tuple[ModelSpec, datetime]] = []
    for m in models:
        complete = have(m, start, end)
        attempts = last_attempt(m, start, end)
        delay = timedelta(hours=availability_delay_h(m))
        for d in range(lookback_days + 1):
            day = (now - timedelta(days=d)).date()
            for hh in m.inits:
                init = datetime(day.year, day.month, day.day, hh, tzinfo=UTC)
                if init + delay > now:
                    continue
                if any(abs((h.to_pydatetime() - init).total_seconds()) < 1 for h in complete):
                    continue
                seen = attempts.get(init)
                if seen is not None and (now - seen.to_pydatetime()) < timedelta(hours=min_retry_h):
                    log.debug("%s %s: partial, retried at %s — waiting", m.model_id, init, seen)
                    continue
                jobs.append((m, init))
    return sorted(jobs, key=lambda j: (j[1], j[0].model_id))


# --------------------------------------------------------------------------- commands


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr")):
    """CastCheck — independent, daily verification of public weather forecasts."""
    setup_logging(verbose)


@app.command()
def version():
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def fetch(model: str = typer.Option(..., help="model_id"), init: str = typer.Option(..., help="e.g. 2026-08-30T00"),
          stations: str = typer.Option("", help="comma-separated ICAO subset")):
    """Fetch one model run and upsert station values."""
    with _journal("fetch") as summary:
        m = model_by_id(model)
        ids = [s for s in stations.split(",") if s] or None
        mid, present, missing = _fetch_one(m, _parse_init(init), ids)
        typer.echo(f"{mid} {init}: present={present} missing={missing}")
        summary.append(f"{mid} {init} present={present} missing={missing}")


@app.command("fetch-latest")
def fetch_latest(lookback_days: int = 3, workers: int = 3,
                 models: str = typer.Option("", help="comma-separated model_ids"),
                 min_retry_h: float = typer.Option(DEFAULT_MIN_RETRY_H,
                                                   help="do not re-attempt an incomplete run more often than this"),
                 fail_on_no_data: bool = typer.Option(True, help="exit 1 when every planned run failed")):
    """Fetch every run that should be available by now (last `lookback_days`) and is not stored yet.

    Exit code 0 unless **every** planned run came back empty, which is the signature of a broken
    pipeline rather than of one late upstream file.
    """
    with _journal("fetch-latest") as summary:
        from .store import existing_inits, last_attempt_by_init

        wanted = [m for m in load_models() if not models or m.model_id in models.split(",")]
        jobs = plan_runs(
            wanted, _now(), lookback_days,
            have=lambda m, s, e: existing_inits(m.model_id, start=s, end=e),
            last_attempt=lambda m, s, e: last_attempt_by_init(m.model_id, start=s, end=e),
            min_retry_h=min_retry_h,
        )
        if not jobs:
            typer.echo("nothing to fetch")
            summary.append("nothing to fetch")
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
                        log.warning("%s %s: no data (upstream late or unavailable)", mid, init.isoformat())
                except Exception as e:  # noqa: BLE001 — never let one run kill the batch
                    failures += 1
                    typer.echo(f"  {mid} {init:%Y-%m-%dT%H}Z ERROR {type(e).__name__}: {e}", err=True)
                    log.exception("%s %s failed", mid, init.isoformat())
        summary.append(f"{len(jobs) - failures}/{len(jobs)} run(s) with data")
        if failures:
            typer.echo(f"{failures} run(s) had no data (will be retried by later runs / backfill)", err=True)
        if failures == len(jobs) and fail_on_no_data:
            raise typer.Exit(1)


@app.command()
def backfill(model: str, start: str, end: str, workers: int = 2, skip_existing: bool = True):
    """Fetch all inits of a model between two dates (inclusive), e.g. --start 2025-01-01 --end 2025-01-31."""
    with _journal("backfill") as summary:
        from .store import existing_inits

        m = model_by_id(model)
        have = existing_inits(m.model_id, start=start, end=end) if skip_existing else set()
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
                    log.exception("%s %s failed", m.model_id, init.isoformat())
        typer.echo(f"done: {ok}/{len(inits)} runs with data")
        summary.append(f"{m.model_id} {start}..{end}: {ok}/{len(inits)} runs with data")


@app.command()
def truth(day: str = typer.Option("", "--date", help="climatological date YYYY-MM-DD (default: UTC yesterday)")):
    """Fetch the final NWS CLI report (with CF6/obs fallback and QC) for all stations for one day."""
    with _journal("truth") as summary:
        from .store import upsert_truth
        from .truth import truth_for_date

        climo = date.fromisoformat(day) if day else (_now().date() - timedelta(days=1))
        rows = truth_for_date(load_stations(), climo)
        out = upsert_truth(rows)
        n_cli = int(((rows["source"] == "CLI") & rows["is_final"]).sum()) if len(rows) else 0
        typer.echo(f"{climo}: {len(rows)} rows ({n_cli} final CLI) -> {out}")
        summary.append(f"{climo}: {len(rows)} rows, {n_cli} final CLI")


@app.command("truth-backfill")
def truth_backfill(start: str, end: str, stations: str = ""):
    """Historic truth from the IEM AFOS archive (CLI) with CF6 filling the gaps."""
    with _journal("truth-backfill") as summary:
        from .store import upsert_truth
        from .truth import truth_backfill as _tb

        sts = load_stations()
        if stations:
            sts = [s for s in sts if s.id in stations.split(",")]
        rows = _tb(sts, date.fromisoformat(start), date.fromisoformat(end))
        typer.echo(f"{len(rows)} rows -> {upsert_truth(rows)}")
        summary.append(f"{start}..{end}: {len(rows)} rows, {len(sts)} station(s)")


def _instant_summary(rows) -> str:
    n = len(rows)
    if not n:
        return "0 instants"
    n_val = int(rows["temp_c"].notna().sum())
    flags = rows.loc[rows["qc_flag"] != "", "qc_flag"].value_counts().to_dict()
    flag_txt = " ".join(f"{k}={v}" for k, v in sorted(flags.items()))
    return f"{n} instants, {n_val} with a value ({100 * n_val / n:.1f}%)" + (f", {flag_txt}" if flag_txt else "")


@app.command("truth-instant")
def truth_instant(day: str = typer.Option("", "--date", help="UTC date YYYY-MM-DD (default: UTC yesterday)"),
                  stations: str = typer.Option("", help="comma-separated ICAO subset")):
    """Observed 2 m temperature at 00/06/12/18 UTC for one day, from api.weather.gov.

    This is the recent-days path: the IEM ASOS archive lags real time, so the daily pipeline fills
    the last week from the live API and ``truth-instant-backfill`` overwrites it with the archive
    later (a value always beats a gap, and the archive always beats the API).
    """
    with _journal("truth-instant") as summary:
        from .store import upsert_truth_instant
        from .truth_instant import truth_instant_for_day

        d = date.fromisoformat(day) if day else (_now().date() - timedelta(days=1))
        sts = _station_subset(stations)
        rows = truth_instant_for_day(sts, d)
        out = upsert_truth_instant(rows)
        typer.echo(f"{d}: {_instant_summary(rows)} -> {out}")
        summary.append(f"{d}: {_instant_summary(rows)}")


@app.command("truth-instant-backfill")
def truth_instant_backfill(start: str, end: str, stations: str = "", workers: int = 3):
    """Observed 2 m temperature at 00/06/12/18 UTC for a date range, from the IEM ASOS archive."""
    with _journal("truth-instant-backfill") as summary:
        from .store import upsert_truth_instant
        from .truth_instant import coverage, truth_instant_for_range

        sts = _station_subset(stations)
        rows = truth_instant_for_range(sts, date.fromisoformat(start), date.fromisoformat(end),
                                       max_workers=workers)
        out = upsert_truth_instant(rows)
        typer.echo(f"{start}..{end}: {_instant_summary(rows)} -> {out}")
        for r in coverage(rows).itertuples():
            typer.echo(f"  {r.station_id} {r.year}: {r.coverage:.4f} "
                       f"(no_report={r.no_report} gap={r.gap_gt35min} suspect={r.suspect})")
        summary.append(f"{start}..{end}: {_instant_summary(rows)}, {len(sts)} station(s)")


def _station_subset(stations: str):
    sts = load_stations()
    return [s for s in sts if s.id in stations.split(",")] if stations else sts


@app.command()
def derive(
    since: int = typer.Option(14, "--since", help="re-derive the runs initialised in the last N days"),
    full: bool = typer.Option(False, "--full", help="re-derive the whole archive instead"),
):
    """Recompute daily_forecasts from forecast_values (idempotent).

    The default is incremental: only the shards whose initialisation date falls in the last
    ``--since`` days are opened, which keeps the daily pipeline O(window) instead of O(archive).
    ``--full`` reproduces the whole table and needs memory proportional to the whole archive.
    """
    with _journal("derive") as summary:
        from .derive import daily_from_values, derive_window
        from .store import read_forecast_values, read_truth_instant, write_daily

        if full:
            values = read_forecast_values()
            daily = daily_from_values(values, load_stations(), load_models(),
                                      truth_instant=read_truth_instant())
            scope = f"{len(values)} values (full)"
        else:
            end = _now().date()
            start = end - timedelta(days=int(since))
            daily = derive_window(start, end, load_stations(), load_models())
            scope = f"inits {start}..{end}"
        typer.echo(f"{scope} -> {len(daily)} daily rows -> {write_daily(daily)}")
        summary.append(f"{scope} -> {len(daily)} daily rows")


@app.command()
def verify(n_boot: int = 1000):
    """Compute scores and pairwise comparisons from daily_forecasts + truth."""
    with _journal("verify") as summary:
        from .derive import instant_errors
        from .store import (
            read_daily,
            read_forecast_values,
            read_truth,
            read_truth_instant,
            write_scores,
        )
        from .verify import score

        daily = read_daily()
        tr = read_truth()
        ti = read_truth_instant()
        instant = None
        if len(ti):
            from .derive import DERIVE_VALUE_COLUMNS

            values = read_forecast_values(columns=DERIVE_VALUE_COLUMNS)
            instant = instant_errors(values, ti, load_stations(), load_models())
            del values
        scores, pairwise = score(daily, tr, instant, n_boot=n_boot, truth_instant=ti)
        as_of = _now().strftime("%Y-%m-%d")
        write_scores(scores, pairwise, as_of)
        typer.echo(f"scores={len(scores)} pairwise={len(pairwise)} as_of={as_of}")
        summary.append(f"scores={len(scores)} pairwise={len(pairwise)} as_of={as_of}")


@app.command("build-site")
def build_site():
    """Generate public/ (HTML + JSON API + status)."""
    with _journal("build-site") as summary:
        from .api import export_api
        from .site.build import build_site as _build
        from .status import write_status
        from .store import read_scores

        scores, pairwise = read_scores()
        export_api(scores, pairwise, load_stations(), load_models())
        write_status(_now())
        _build(_now())
        typer.echo(f"site written to {PUBLIC_DIR}")
        summary.append(f"site written to {PUBLIC_DIR}")


@app.command()
def status(fail_on_gaps: bool = True):
    """Write public/api/v1/status.json; exit 1 if today's runs or truth are incomplete."""
    with _journal("status") as summary:
        from .status import write_status

        rep = write_status(_now())
        gaps = rep.get("gaps_today", [])
        typer.echo(f"gaps today: {len(gaps)}")
        for g in gaps[:50]:
            typer.echo(f"  {g}")
        summary.append(f"{len(gaps)} gap(s) today")
        if gaps and fail_on_gaps:
            raise typer.Exit(1)


@app.command()
def daily(n_boot: int = 1000):
    """derive -> verify -> build-site in one go (used by the daily workflow).

    Each step journals itself as well, so `last_run.json` shows both the whole pipeline and which
    stage it stopped at.
    """
    with _journal("daily") as summary:
        derive()
        verify(n_boot=n_boot)
        build_site()
        summary.append("derive + verify + build-site")


@publish_app.command("hf")
def publish_hf(repo: str = "castcheck/temperature-verification", private: bool = False,
               dry_run: bool = typer.Option(False, help="report what would be pushed, upload nothing")):
    """Push data/ to a Hugging Face dataset repo (needs HF_TOKEN)."""
    with _journal("publish-hf") as summary:
        from .publish.hf import push_dataset

        out = push_dataset(repo, private=private, dry_run=dry_run)
        typer.echo(out)
        summary.append(out.splitlines()[0] if out else "")


@publish_app.command("kaggle")
def publish_kaggle(slug: str = "castcheck-temperature-forecast-verification",
                   dry_run: bool = typer.Option(False, help="stage the folder, run no Kaggle command")):
    """Mirror the published tables to Kaggle (needs KAGGLE_API_TOKEN)."""
    with _journal("publish-kaggle") as summary:
        from .publish.kaggle import push_dataset

        out = push_dataset(slug, dry_run=dry_run)
        typer.echo(out)
        summary.append(out.splitlines()[-1] if out else "")


@publish_app.command("bluesky")
def publish_bluesky(dry_run: bool = False):
    """Post the daily chart to Bluesky (needs BSKY_HANDLE + BSKY_APP_PASSWORD)."""
    with _journal("publish-bluesky") as summary:
        from .publish.bluesky import post_daily

        out = post_daily(dry_run=dry_run)
        typer.echo(out)
        summary.append(out.splitlines()[0] if out else "")


if __name__ == "__main__":
    app()
