"""Which completeness gaps have been overdue for more than a day (DESIGN §7, `health.yml`).

`castcheck status` answers "is anything missing *right now*", which is the wrong question for an
alert: at 12:30 UTC a 12Z run is legitimately absent, and a fetch that fails once is picked up by
the next pass an hour later. What deserves a human is a slot whose publication deadline passed a
whole day ago and that is *still* empty — the pipeline has had four or more chances at it.

The report's per-day grids are the input, not `current_gaps`: `current_gaps` only lists today (and,
for truth, yesterday), while a stuck slot from four days ago is exactly what this looks for. Each
grid row already carries the `due_at` that `castcheck.schedule` computed, so "how long overdue" is
read off the data and needs no state file between runs.

    PYTHONPATH=. .venv/bin/python scripts/health_gaps.py [--status-json public/api/v1/status.json]
                                                         [--max-age-h 24] [--now 2026-09-01T12:30:00Z]

Exit code 0 when nothing is overdue, 2 when something is; anything else is a real error. Writes a
GitHub-flavoured Markdown summary to stdout (the issue body) and, with `--github-output`, the
`overdue=<n>` line the workflow branches on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATUS = REPO_ROOT / "public" / "api" / "v1" / "status.json"
DEFAULT_MAX_AGE_H = 24.0
#: Only slots whose deadline passed within this many days can raise the alarm. Older holes are a
#: *backfill* question, not a "the pipeline has stopped" question: an in-flight backfill leaves
#: months of legitimately-empty slots behind it, and an alert that includes them would be red on the
#: day it is switched on and stay red forever, which is the same as having no alert. They are still
#: counted in the issue body so the number never silently disappears.
DEFAULT_LOOKBACK_DAYS = 7.0
#: The issue body lists at most this many rows; the count above it is always the true total.
MAX_LISTED = 40


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _instant_due_at(day: str) -> datetime | None:
    """The observed-instant deadline, from `castcheck.schedule` so the two never disagree.

    `truth_instant` day rows are the one grid that does not publish `due_at`. Importing keeps this
    script honest; when castcheck is not importable (someone running the script from a tarball) the
    row is reported without an age rather than silently dropped.
    """
    try:
        from castcheck import schedule
    except ImportError:  # pragma: no cover - only outside the repo venv
        return None
    return schedule.instant_due_at(date.fromisoformat(day))


def overdue_gaps(report: dict, now: datetime, max_age_h: float = DEFAULT_MAX_AGE_H) -> list[dict]:
    """Every expected-but-missing slot whose deadline passed more than `max_age_h` ago.

    Pure: `report` is a parsed status.json, `now` the instant to measure against.
    """
    out: list[dict] = []

    def add(kind: str, what: str, day: str, due_at: datetime | None, detail: str) -> None:
        if due_at is None:
            return
        age_h = (now - due_at).total_seconds() / 3600.0
        if age_h > max_age_h:
            out.append({"type": kind, "what": what, "date": day, "due_at": due_at.isoformat(),
                        "age_h": round(age_h, 1), "detail": detail})

    for m in report.get("models", []):
        for d in m.get("days", []):
            if d.get("expected") and not d.get("complete"):
                add("model_run", f"{m['model_id']} {int(m['init_hour']):02d}Z", d["date"],
                    _parse_dt(d["due_at"]) if d.get("due_at") else None,
                    f"{d.get('stations_complete', 0)}/{report.get('n_stations', '?')} stations with "
                    f"{d.get('expected_steps', '?')} steps")
    for t in report.get("truth", []):
        for d in t.get("days", []):
            if d.get("expected") and not d.get("cli_final"):
                add("truth", t["station_id"], d["date"],
                    _parse_dt(d["due_at"]) if d.get("due_at") else None,
                    "no first-final CLI with both tmax and tmin")
    for t in report.get("truth_instant", []):
        for d in t.get("days", []):
            if d.get("expected") and not d.get("complete"):
                add("truth_instant", t["station_id"], d["date"], _instant_due_at(d["date"]),
                    f"{d.get('n_instants', 0)}/4 synoptic instants observed")

    out.sort(key=lambda g: (-g["age_h"], g["type"], g["what"]))
    return out


def split_recent(rows: list[dict], lookback_days: float = DEFAULT_LOOKBACK_DAYS,
                 ) -> tuple[list[dict], list[dict]]:
    """`(alerting, archive)` — slots overdue inside the lookback window, and the older ones."""
    cut = float(lookback_days) * 24.0
    return [r for r in rows if r["age_h"] <= cut], [r for r in rows if r["age_h"] > cut]


def markdown(rows: list[dict], archive: list[dict], report: dict, now: datetime,
             max_age_h: float, lookback_days: float) -> str:
    """The issue body: worst-first table plus the context needed to judge it without a checkout."""
    head = (f"`castcheck status` for **{report.get('as_of', '?')}** (checked "
            f"{now.replace(microsecond=0).isoformat()}) — schema v{report.get('schema_version', '?')}, "
            f"castcheck v{report.get('castcheck_version', '?')}.\n")
    tail = (f"\n{len(archive)} older slot(s) are also incomplete but their deadline passed more than "
            f"{lookback_days:g} days ago — backfill territory, not a stalled pipeline, so they do not "
            "raise this alarm." if archive else "")
    if not rows:
        return (head + f"\nNo slot due in the last {lookback_days:g} days has been overdue by more "
                f"than {max_age_h:g} h. The daily pipeline is keeping up." + tail)
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    lines = [
        head,
        f"\n**{len(rows)} slot(s) overdue by more than {max_age_h:g} h** "
        f"({', '.join(f'{v} {k}' for k, v in sorted(by_type.items()))}). "
        "Every one of these has had at least four scheduled passes at it.\n",
        "\n| overdue | kind | what | date | detail |",
        "|---:|---|---|---|---|",
    ]
    for r in rows[:MAX_LISTED]:
        lines.append(f"| {r['age_h']:.0f} h | {r['type']} | `{r['what']}` | {r['date']} | {r['detail']} |")
    if len(rows) > MAX_LISTED:
        lines.append(f"\n…and {len(rows) - MAX_LISTED} more.")
    lines += [
        tail,
        "\nFull picture: <https://castcheck.zifanzhang.com/status/> · "
        "`public/api/v1/status.json`.",
        "\nThis issue is closed automatically by `health.yml` as soon as nothing is overdue; "
        "it is updated in place, so the comment thread is the history of the outage.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status-json", default=str(DEFAULT_STATUS), help="path to status.json")
    ap.add_argument("--max-age-h", type=float, default=DEFAULT_MAX_AGE_H,
                    help="how long past its deadline a slot may stay empty (default: 24)")
    ap.add_argument("--lookback-days", type=float, default=DEFAULT_LOOKBACK_DAYS,
                    help="only slots due within this many days can raise the alarm (default: 7)")
    ap.add_argument("--now", default="", help="ISO instant to measure against (default: the clock)")
    ap.add_argument("--github-output", action="store_true",
                    help="also append overdue=<n> to $GITHUB_OUTPUT")
    args = ap.parse_args(argv)

    path = Path(args.status_json)
    if not path.exists():
        print(f"status.json not found at {path}; run `castcheck status --no-fail-on-gaps` first",
              file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    now = _parse_dt(args.now) if args.now else datetime.now(UTC)
    rows, archive = split_recent(overdue_gaps(report, now, args.max_age_h), args.lookback_days)
    print(markdown(rows, archive, report, now, args.max_age_h, args.lookback_days))

    out = os.environ.get("GITHUB_OUTPUT")
    if args.github_output and out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"overdue={len(rows)}\n")
    return 2 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
