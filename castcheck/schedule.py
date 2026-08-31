"""When a thing is *due* — the single source of truth for upstream availability delays.

Two commands need the same answer to "should this exist by now?" and they used to answer it
separately: ``cli.plan_runs`` decides which model runs are worth fetching, and ``status.build``
decides which absent runs are a *gap* rather than a run that has simply not been published yet.
Keeping two copies of the delays guarantees that the fetcher and the status page eventually
disagree — the page would show a red bar for a run the fetcher correctly has not asked for.  So the
constants and the arithmetic live here, and both import them.

Three kinds of deadline:

* **model runs** — ``init + availability_delay``.  The delay is per source, and for AIWP per
  *initial field*: NOAA/CIRA has to wait for ECMWF's dissemination before it can run the
  IFS-initialised models, so those appear about three and a half hours after the GFS-initialised
  ones (DESIGN §7).
* **CLI truth** — the first final NWS Daily Climate Report is issued *after local midnight*, and in
  practice one to four hours after it.  The deadline is therefore per station (its fixed standard
  offset) and per climatological day, not a flat "yesterday".  At 06 UTC, yesterday's report exists
  for KNYC and does not yet exist for KLAX.
* **observed instants** — the last synoptic instant of a UTC day is 18 UTC, plus the time it takes
  the routine METAR to reach the archive.

Everything takes an explicit ``now`` so that the callers and the tests are deterministic.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

__all__ = [
    "AIWP_DELAY_H_BY_INIT_FIELD",
    "AVAILABILITY_DELAY_H",
    "DEFAULT_DELAY_H",
    "INSTANT_DELAY_H",
    "TRUTH_CLI_DELAY_H",
    "availability_delay_h",
    "instant_due_at",
    "instant_is_due",
    "now_utc",
    "run_due_at",
    "run_is_due",
    "truth_due_at",
    "truth_is_due",
]

#: Expected availability delay (hours after init) before a run is worth fetching; measured
#: 2026-08-30.  Keyed by ``ModelSpec.source``; the same delay applies to the 00Z and 12Z cycles.
AVAILABILITY_DELAY_H = {"ecmwf": 8.0, "gfs": 5.5, "aiwp": 9.5}

#: AIWP publishes the GFS-initialised runs about three hours before the IFS-initialised ones,
#: because it has to wait for ECMWF's own dissemination first.
AIWP_DELAY_H_BY_INIT_FIELD = {"GFS": 6.0, "IFS": 9.5}

#: Fallback for a source with no measured delay: the slowest one we know about.
DEFAULT_DELAY_H = 8.0

#: Hours after the station's *local* midnight before the first final CLI can be expected.  The
#: products are issued one to four hours after midnight LST; the deadline is the late end, because
#: a status page that turns red on a report that is merely punctual is worse than one that waits.
TRUTH_CLI_DELAY_H = 4.0

#: Hours after 18 UTC before the day's fourth synoptic observation should have reached the archive.
INSTANT_DELAY_H = 1.5


def now_utc() -> datetime:
    """The current instant, UTC, seconds resolution."""
    return datetime.now(UTC).replace(microsecond=0)


def availability_delay_h(model) -> float:
    """Hours after initialisation at which ``model``'s run is normally complete upstream."""
    if getattr(model, "source", None) == "aiwp":
        return AIWP_DELAY_H_BY_INIT_FIELD.get(
            getattr(model, "init_field", None) or "GFS", AVAILABILITY_DELAY_H["aiwp"])
    return AVAILABILITY_DELAY_H.get(getattr(model, "source", None), DEFAULT_DELAY_H)


def run_due_at(model, init_time: datetime) -> datetime:
    """The instant at which ``model``'s ``init_time`` run should be complete upstream."""
    init = init_time if init_time.tzinfo is not None else init_time.replace(tzinfo=UTC)
    return init + timedelta(hours=availability_delay_h(model))


def run_is_due(model, init_time: datetime, now: datetime | None = None) -> bool:
    """``True`` when the run should already exist, i.e. its absence is a gap and not a wait."""
    return run_due_at(model, init_time) <= (now or now_utc())


def truth_due_at(station, climo_date: date) -> datetime:
    """The instant at which ``station``'s first final CLI for ``climo_date`` should exist.

    Local midnight *after* the climatological day, in the station's fixed standard offset, plus
    :data:`TRUTH_CLI_DELAY_H`.
    """
    offset = int(getattr(station, "std_offset_h", 0) or 0)
    midnight_utc = datetime(climo_date.year, climo_date.month, climo_date.day,
                            tzinfo=UTC) + timedelta(days=1, hours=-offset)
    return midnight_utc + timedelta(hours=TRUTH_CLI_DELAY_H)


def truth_is_due(station, climo_date: date, now: datetime | None = None) -> bool:
    """``True`` when the first final CLI for that station-day should already have been issued."""
    return truth_due_at(station, climo_date) <= (now or now_utc())


def instant_due_at(day: date) -> datetime:
    """The instant at which all four synoptic observations of a UTC day should be archived."""
    return datetime(day.year, day.month, day.day, 18, tzinfo=UTC) + timedelta(
        hours=INSTANT_DELAY_H)


def instant_is_due(day: date, now: datetime | None = None) -> bool:
    """``True`` when the day's four observed instants should already be in the archive."""
    return instant_due_at(day) <= (now or now_utc())
