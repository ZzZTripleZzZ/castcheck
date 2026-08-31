"""Configuration loading: stations and models (the only readers of config/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
PUBLIC_DIR = REPO_ROOT / "public"

USER_AGENT = "castcheck/0.1 (https://castcheck.zifanzhang.com; zhangzifan716@gmail.com)"


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    cli_pil: str
    tz: str
    std_offset_h: int
    lat: float | None
    lon: float | None
    elev_m: float | None
    kalshi: str | None = None

    @property
    def cli_location(self) -> str:
        """api.weather.gov location code for the CLI product (pil without the CLI prefix)."""
        return self.cli_pil[3:]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    source: str
    product: str
    init_field: str | None
    inits: tuple[int, ...]
    step_h: int
    max_h: int
    native_extremes: tuple[str, ...]


def standard_offset_hours(tz: str) -> int:
    """Fixed *standard* UTC offset of an IANA zone (the smaller of the January/July offsets).

    NWS climatological days are defined in local standard time all year, so DST must be ignored.
    """
    z = ZoneInfo(tz)
    jan = datetime(2025, 1, 15, 12, tzinfo=z).utcoffset()
    jul = datetime(2025, 7, 15, 12, tzinfo=z).utcoffset()
    assert jan is not None and jul is not None
    off = min(jan, jul)  # northern hemisphere: standard time is the smaller (more negative) offset
    hours = off.total_seconds() / 3600
    if hours != int(hours):
        raise ValueError(f"non-integer standard offset for {tz}: {hours}")
    return int(hours)


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_stations() -> list[Station]:
    raw = _load_yaml("stations.yaml")["stations"]
    out: list[Station] = []
    for s in raw:
        out.append(
            Station(
                id=s["id"],
                name=s["name"],
                cli_pil=s["cli_pil"],
                tz=s["tz"],
                std_offset_h=int(s.get("std_offset_h", standard_offset_hours(s["tz"]))),
                lat=s.get("lat"),
                lon=s.get("lon"),
                elev_m=s.get("elev_m"),
                kalshi=s.get("kalshi"),
            )
        )
    ids = [s.id for s in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate station ids in stations.yaml")
    return out


def station_by_id(station_id: str) -> Station:
    for s in load_stations():
        if s.id == station_id:
            return s
    raise KeyError(station_id)


@lru_cache(maxsize=1)
def load_models() -> list[ModelSpec]:
    raw = _load_yaml("models.yaml")["models"]
    out: list[ModelSpec] = []
    for m in raw:
        out.append(
            ModelSpec(
                model_id=m["model_id"],
                family=m["family"],
                source=m["source"],
                product=m["product"],
                init_field=m.get("init_field"),
                inits=tuple(int(h) for h in m.get("inits", (0, 12))),
                step_h=int(m.get("step_h", 6)),
                max_h=int(m.get("max_h", 240)),
                native_extremes=tuple(m.get("native_extremes", ())),
            )
        )
    ids = [m.model_id for m in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model ids in models.yaml")
    return out


def model_by_id(model_id: str) -> ModelSpec:
    for m in load_models():
        if m.model_id == model_id:
            return m
    raise KeyError(model_id)


def baseline_ids() -> list[str]:
    return [b["model_id"] for b in _load_yaml("models.yaml").get("baselines", [])]


@lru_cache(maxsize=1)
def display_names() -> dict[str, str]:
    """``model_id -> human label``, e.g. ``graphcast_ifs -> "GraphCast (IFS init)"``.

    Covers the baselines too, so anything that renders a ``model_id`` (site, charts, posts) shows
    the same name. Unknown ids fall back to the id itself.
    """
    out: dict[str, str] = {}
    for m in load_models():
        out[m.model_id] = f"{m.family} ({m.init_field} init)" if m.init_field else m.family
    for b in _load_yaml("models.yaml").get("baselines", []):
        out[b["model_id"]] = b.get("family", b["model_id"])
    return out


def display_name(model_id: str) -> str:
    """Human label for one ``model_id`` (see :func:`display_names`)."""
    return display_names().get(model_id, model_id)
