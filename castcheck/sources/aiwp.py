"""AIWP adapter — NOAA/CIRA operational AI weather prediction reforecasts (DESIGN §4, §9 package C).

Data source
-----------
Public AWS Open Data bucket ``noaa-oar-mlwp-data`` (anonymous HTTPS, no credentials).
Object layout::

    {PROD}_{ver}_{INIT}/{YYYY}/{MMDD}/{PROD}_{ver}_{INIT}_{YYYYMMDDHH}_f000_f240_06.nc

e.g. ``GRAP_v100_IFS/2026/0830/GRAP_v100_IFS_2026083000_f000_f240_06.nc``.
``PROD`` ∈ {GRAP, PANG, FOUR, AURO}, ``INIT`` ∈ {GFS, IFS}, ``ver`` ∈ {v100, v200, …} — the
version set is *discovered* by listing the bucket root, because a product may ship two versions
side by side (FourCastNet v1 = ``v100`` and v2-small = ``v200`` overlap in 2020–2023).  The
version actually used for a run is recorded in ``model_version`` as ``"{PROD}_{ver}"``.

Files are 2.8–4.6 GB each and are **never downloaded**: they are opened lazily over HTTP with
``h5netcdf`` on an ``fsspec`` file object.  ``t2`` has shape (41, 721, 1440) — time × latitude ×
longitude — chunked one full layer at a time, so the only efficient access pattern is *read one
time index, then extract every station from that layer* (``grid.extract_all``).  Latitude runs
90 → −90, longitude 0 → 359.75.  The ``f000`` layer is written as the fill value 9.96921e+36 and
is therefore recorded as an explicit ``fill_value`` missing row, never as a temperature.

06/18 UTC initializations exist for 2023 only and are deliberately ignored (METHODOLOGY §1).

A kerchunk reference set exists under ``parquet/{PROD}_{ver}_{INIT}_combined_all.parq/`` but is
not used: its chunking is identical (1, 1, 721, 1440), so it offers no per-gridpoint speedup, and
it was last regenerated 2025-04-01, so it does not cover recent runs.

License / citation
------------------
NOAA Open Data Dissemination; the data are freely available.  If you use this dataset, cite:

    Radford, J. T., I. Ebert-Uphoff, J. Q. Stewart, K. D. Musgrave, R. DeMaria, N. Tourville,
    and K. Hilburn, 2025: Accelerating Community-Wide Evaluation of AI Models for Global Weather
    Prediction by Facilitating Access to Model Output. Bull. Amer. Meteor. Soc., 106, E68–E76,
    https://doi.org/10.1175/BAMS-D-24-0057.1.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd

from .. import grid
from ..config import USER_AGENT, ModelSpec, Station
from . import _http
from .base import FetchRequest, FetchResult, k_to_c, make_rows

log = logging.getLogger(__name__)

BUCKET = "noaa-oar-mlwp-data"
BASE_URL = f"https://{BUCKET}.s3.amazonaws.com/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

PRODUCTS = ("GRAP", "PANG", "FOUR", "AURO")
INIT_FIELDS = ("GFS", "IFS")

#: Bucket-root directory name, e.g. ``GRAP_v100_IFS``.
DIR_RE = re.compile(r"^(?P<prod>GRAP|PANG|FOUR|AURO)_(?P<ver>v\d{3})_(?P<init>GFS|IFS)/$")
#: Object basename, e.g. ``GRAP_v100_IFS_2026083000_f000_f240_06.nc``.
FILE_RE = re.compile(
    r"^(?P<prod>GRAP|PANG|FOUR|AURO)_(?P<ver>v\d{3})_(?P<init>GFS|IFS)_"
    r"(?P<stamp>\d{10})_f(?P<f0>\d{3})_f(?P<f1>\d{3})_(?P<step>\d{2})\.nc$"
)

DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3
# fsspec read-ahead block size. One compressed t2 layer is ~1.7 MB; measured over the 40 forecast
# steps of one GraphCast file: 1 MiB -> 70 MB / 45 s, 2 MiB -> 108 MB / 7.3 s, 4 MiB -> 174 MB / 8.9 s.
DEFAULT_BLOCK_SIZE = 2 << 20

VARIABLE = "t2"
FILL_REASON = "fill_value"


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _retry(fn, *, retries: int = DEFAULT_RETRIES, what: str = "request", notes: list[str] | None = None):
    """Call ``fn`` with exponential backoff. Returns (value, error) — never raises."""
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 — adapters must never raise on partial failure
            last = exc
            msg = f"{what}: attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}"
            log.warning(msg)
            if notes is not None:
                notes.append(msg)
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
    return None, last


# --------------------------------------------------------------------------- naming


def dir_name(product: str, version: str, init_field: str) -> str:
    """``("GRAP", "v100", "IFS") -> "GRAP_v100_IFS"``."""
    return f"{product}_{version}_{init_field}"


def object_key(product: str, version: str, init_field: str, init_time: datetime) -> str:
    """S3 key of the single NetCDF file holding f000..f240 of one initialization."""
    t = _utc(init_time)
    d = dir_name(product, version, init_field)
    return f"{d}/{t:%Y}/{t:%m%d}/{d}_{t:%Y%m%d%H}_f000_f240_06.nc"


def object_url(product: str, version: str, init_field: str, init_time: datetime) -> str:
    return BASE_URL + object_key(product, version, init_field, init_time)


def parse_key(key: str) -> tuple[str, str, str, datetime] | None:
    """Inverse of :func:`object_key`: key -> (product, version, init_field, init_time) or None."""
    m = FILE_RE.match(key.rsplit("/", 1)[-1])
    if not m:
        return None
    t = datetime.strptime(m["stamp"], "%Y%m%d%H").replace(tzinfo=UTC)
    return m["prod"], m["ver"], m["init"], t


def model_version(product: str, version: str) -> str:
    """``model_version`` written into forecast_values, e.g. ``"GRAP_v100"`` (DESIGN §3.1)."""
    return f"{product}_{version}"


def bytes_read(fh) -> int:
    """Bytes actually pulled over HTTP by an fsspec file (0 if the cache does not track it)."""
    return int(getattr(getattr(fh, "cache", None), "total_requested_bytes", 0) or 0)


# --------------------------------------------------------------------------- source


class AiwpSource:
    """Source adapter for ``model.source == "aiwp"`` (DESIGN §4 ``Source`` protocol).

    Parameters
    ----------
    version:
        Pin a specific version directory (``"v100"``/``"v200"``).  By default the highest
        version that actually contains the requested initialization is used, so a product that
        ships two versions (FourCastNet) is fetched from the newer one and the version is
        recorded in ``model_version``.
    """

    def __init__(
        self,
        *,
        version: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        self.version = version
        self.timeout = timeout
        self.retries = retries
        self.block_size = block_size
        self._version_cache: dict[tuple[str, str], list[str]] = {}
        self._listing_cache: dict[str, list[str]] = {}

    # ---------------------------------------------------------------- listing

    def _list(self, prefix: str, *, delimiter: bool) -> tuple[list[str], list[str]] | None:
        """Paged anonymous ListObjectsV2. Returns (keys, common_prefixes) or None if the listing failed."""
        keys: list[str] = []
        prefixes: list[str] = []
        token: str | None = None
        while True:
            url = f"{BASE_URL}?list-type=2&prefix={quote(prefix)}&max-keys=1000"
            if delimiter:
                url += "&delimiter=/"
            if token:
                url += f"&continuation-token={quote(token)}"
            res = _http.fetch(url, timeout=self.timeout, retries=self.retries)
            if not res.ok:
                log.warning("listing %s failed: %s", url, res.reason)
                return None
            try:
                root = ET.fromstring(res.content or b"")
            except ET.ParseError as exc:
                log.warning("listing %s returned unparseable XML: %s", url, exc)
                return None
            keys += [e.text or "" for e in root.iter(f"{S3_NS}Key")]
            for cp in root.iter(f"{S3_NS}CommonPrefixes"):
                p = cp.find(f"{S3_NS}Prefix")
                if p is not None and p.text and p.text != prefix:
                    prefixes.append(p.text)
            nxt = root.find(f"{S3_NS}NextContinuationToken")
            if nxt is None or not nxt.text:
                return keys, prefixes
            token = nxt.text

    def _list_keys(self, prefix: str) -> list[str] | None:
        out = self._list(prefix, delimiter=False)
        return None if out is None else out[0]

    def _list_common_prefixes(self, prefix: str) -> list[str] | None:
        out = self._list(prefix, delimiter=True)
        return None if out is None else out[1]

    # ---------------------------------------------------------------- versions

    def discover_versions(self, product: str, init_field: str) -> list[str]:
        """Version directories present in the bucket for a product × initial field.

        Returns e.g. ``["v100", "v200"]``, ascending; ``[]`` if the combination does not exist
        (``FOUR_v100_IFS`` never existed).  Network failures degrade to ``[]`` rather than raising.
        """
        product = product.upper()
        init_field = init_field.upper()
        cached = self._version_cache.get((product, init_field))
        if cached is not None:
            return cached
        prefixes = self._list_common_prefixes("")
        if prefixes is None:
            log.warning("discover_versions(%s, %s): bucket listing failed", product, init_field)
            return []
        vers = sorted(
            {
                m["ver"]
                for p in prefixes
                if (m := DIR_RE.match(p)) and m["prod"] == product and m["init"] == init_field
            }
        )
        self._version_cache[(product, init_field)] = vers
        return vers

    def _versions_for(self, model: ModelSpec) -> list[str]:
        if self.version:
            return [self.version]
        return list(reversed(self.discover_versions(model.product, model.init_field or "GFS")))

    def resolve_version(self, model: ModelSpec, init_time: datetime) -> str | None:
        """Highest available version directory that actually holds this initialization, else None."""
        for ver in self._versions_for(model):
            url = object_url(model.product, ver, model.init_field or "GFS", init_time)
            res = _http.fetch(url, head=True, timeout=self.timeout, retries=self.retries)
            if res.ok:
                return ver
            if res.reason != "http_404":
                log.warning("HEAD %s: %s", url, res.reason)
        return None

    # ---------------------------------------------------------------- inits

    def available_inits(self, model: ModelSpec, start: date, end: date) -> list[datetime]:
        """Initializations present in the archive within ``[start, end]`` (inclusive dates).

        Only the initialization hours declared by the model (00/12 UTC) are returned; the 2023-only
        06/18 UTC runs are ignored.  The union over all version directories is returned, so a date
        covered by either ``v100`` or ``v200`` counts as available.
        """
        init_field = (model.init_field or "GFS").upper()
        versions = self._versions_for(model)
        wanted_hours = set(model.inits)
        found: set[datetime] = set()
        for ver in versions:
            d = dir_name(model.product, ver, init_field)
            for year in range(start.year, end.year + 1):
                prefix = f"{d}/{year}/"
                keys = self._listing_cache.get(prefix)
                if keys is None:
                    keys = self._list_keys(prefix)
                    if keys is None:
                        continue  # transient listing failure; other versions/years still count
                    self._listing_cache[prefix] = keys
                for k in keys:
                    parsed = parse_key(k)
                    if not parsed:
                        continue
                    _p, _v, _i, t = parsed
                    if t.hour in wanted_hours and start <= t.date() <= end:
                        found.add(t)
        return sorted(found)

    def archive_bounds(self, product: str, init_field: str, version: str) -> tuple[date, date] | None:
        """(earliest, latest) initialization date present for one version directory."""
        d = dir_name(product, version, init_field)
        year_prefixes = self._list_common_prefixes(f"{d}/")
        if not year_prefixes:
            return None
        years = sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in year_prefixes)
        first_days = sorted(self._list_common_prefixes(f"{d}/{years[0]}/") or [])
        last_days = sorted(self._list_common_prefixes(f"{d}/{years[-1]}/") or [])
        if not first_days or not last_days:
            return None
        lo = date.fromisoformat(years[0] + first_days[0].rstrip("/").rsplit("/", 1)[-1])
        hi = date.fromisoformat(years[-1] + last_days[-1].rstrip("/").rsplit("/", 1)[-1])
        return lo, hi

    # ---------------------------------------------------------------- fetching

    def _open(self, url: str):
        """Lazily open the remote NetCDF over HTTP; returns (h5netcdf.File, fsspec file)."""
        import fsspec
        import h5netcdf

        fs = fsspec.filesystem(
            "http",
            block_size=self.block_size,
            client_kwargs={"headers": {"User-Agent": USER_AGENT}},
        )
        fh = fs.open(url, "rb")
        ds = h5netcdf.File(fh, "r")
        return ds, fh

    def expected_valid_times(self, model: ModelSpec, init_time: datetime) -> list[datetime]:
        """f000 … f{max_h} at ``model.step_h``; f000 is included so its fill value is recorded."""
        init = _utc(init_time)
        return [init + timedelta(hours=h) for h in range(0, model.max_h + 1, model.step_h)]

    def fetch_run(self, req: FetchRequest) -> FetchResult:
        """Extract 2 m temperature for every station and every forecast step of one run.

        Never raises: a missing object, a network failure or a fill-value layer produces explicit
        rows with ``missing_reason`` (``http_404``, ``timeout``, ``no_field``, ``fill_value``).
        """
        model = req.model
        if model.source != "aiwp":
            raise ValueError(f"AiwpSource cannot handle source={model.source!r}")
        init_time = _utc(req.init_time)
        stations = req.stations
        notes: list[str] = []
        valid_times = self.expected_valid_times(model, init_time)
        init_field = (model.init_field or "GFS").upper()

        ver = self.resolve_version(model, init_time)
        if ver is None:
            candidates = self._versions_for(model)
            url = (
                object_url(model.product, candidates[0], init_field, init_time)
                if candidates
                else BASE_URL + dir_name(model.product, "v???", init_field)
            )
            notes.append(f"no object for {model.model_id} {init_time:%Y-%m-%dT%H}Z (tried {candidates})")
            return FetchResult(
                rows=self._missing_rows(model, "unknown", init_time, valid_times, stations, "http_404", url),
                notes=notes,
            )

        url = object_url(model.product, ver, init_field, init_time)
        mver = model_version(model.product, ver)

        opened, err = _retry(lambda: self._open(url), retries=self.retries, what=f"open {url}", notes=notes)
        if opened is None:
            notes.append(f"could not open {url}: {err}")
            return FetchResult(
                rows=self._missing_rows(model, mver, init_time, valid_times, stations, "timeout", url),
                notes=notes,
            )
        ds, fh = opened

        try:
            frames = self._read_all_layers(ds, model, mver, init_time, valid_times, stations, url, notes)
        finally:
            n_bytes = bytes_read(fh)
            try:
                ds.close()
            finally:
                fh.close()

        notes.append(f"bytes_read={n_bytes}")
        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return FetchResult(rows=rows, notes=notes)

    # ---------------------------------------------------------------- internals

    def _missing_rows(
        self,
        model: ModelSpec,
        mver: str,
        init_time: datetime,
        valid_times: Iterable[datetime],
        stations: list[Station],
        reason: str,
        url: str,
    ) -> pd.DataFrame:
        frames = [
            make_rows(
                model=model,
                model_version=mver,
                init_time=init_time,
                valid_time=vt,
                lead_h=int((vt - init_time).total_seconds() // 3600),
                variable=VARIABLE,
                bucket_h=0,
                source_url=url,
                values={},
                stations=stations,
                missing_reason=reason,
            )
            for vt in valid_times
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _read_all_layers(
        self,
        ds,
        model: ModelSpec,
        mver: str,
        init_time: datetime,
        valid_times: list[datetime],
        stations: list[Station],
        url: str,
        notes: list[str],
    ) -> list[pd.DataFrame]:
        if VARIABLE not in ds.variables:
            notes.append(f"{url}: no variable {VARIABLE!r}")
            return [self._missing_rows(model, mver, init_time, valid_times, stations, "no_field", url)]

        var = ds.variables[VARIABLE]
        lats = np.asarray(ds.variables["latitude"][:], dtype=float)
        lons = np.asarray(ds.variables["longitude"][:], dtype=float)
        file_times = self._file_valid_times(ds, notes)
        n_steps = var.shape[0]

        frames: list[pd.DataFrame] = []
        for i, vt in enumerate(valid_times):
            lead_h = int((vt - init_time).total_seconds() // 3600)
            if i >= n_steps:
                notes.append(f"{url}: step f{lead_h:03d} beyond file (n={n_steps})")
                frames.append(self._missing_rows(model, mver, init_time, [vt], stations, "no_field", url))
                continue
            if file_times is not None and file_times[i] != vt:
                notes.append(f"{url}: time[{i}]={file_times[i]:%Y-%m-%dT%H}Z != expected {vt:%Y-%m-%dT%H}Z")
                frames.append(
                    self._missing_rows(model, mver, init_time, [vt], stations, "time_mismatch", url)
                )
                continue

            # float64 up front: grid.bilinear/nearest each call np.asarray(field, dtype=float),
            # which would otherwise re-convert the whole 721x1440 layer once per station per method.
            layer, err = _retry(
                lambda idx=i: np.asarray(var[idx], dtype=float),
                retries=self.retries,
                what=f"{url} layer {i}",
            )
            if layer is None:
                notes.append(f"{url}: layer {i} unreadable: {err}")
                frames.append(self._missing_rows(model, mver, init_time, [vt], stations, "timeout", url))
                continue

            raw = grid.extract_all(layer, lats, lons, stations)
            values: dict[str, tuple[float | None, float | None]] = {}
            fill_ids: list[str] = []
            for sid, (bl_k, nn_k) in raw.items():
                if grid.is_fill(bl_k) or grid.is_fill(nn_k):
                    fill_ids.append(sid)
                    continue
                values[sid] = (k_to_c(bl_k), k_to_c(nn_k))
            reason = FILL_REASON if fill_ids else "no_value"
            if fill_ids and len(fill_ids) == len(raw):
                notes.append(f"{url}: f{lead_h:03d} is fill value for all stations")
            frames.append(
                make_rows(
                    model=model,
                    model_version=mver,
                    init_time=init_time,
                    valid_time=vt,
                    lead_h=lead_h,
                    variable=VARIABLE,
                    bucket_h=0,
                    source_url=url,
                    values=values,
                    stations=stations,
                    missing_reason=reason,
                )
            )
        return frames

    @staticmethod
    def _file_valid_times(ds, notes: list[str]) -> list[datetime] | None:
        """Valid times from the file's ``time`` variable (seconds since 1970), or None."""
        if "time" not in ds.variables:
            notes.append("file has no 'time' variable; trusting init + n*step_h")
            return None
        try:
            raw = np.asarray(ds.variables["time"][:]).astype("int64")
            units = ds.variables["time"].attrs.get("units", "seconds since 1970-1-1")
            if isinstance(units, bytes):
                units = units.decode()
            if "second" not in str(units):
                notes.append(f"unexpected time units {units!r}; trusting init + n*step_h")
                return None
            return [datetime.fromtimestamp(int(s), tz=UTC) for s in raw]
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not decode time variable: {exc}")
            return None


def fetch_many(reqs: list[FetchRequest], *, max_workers: int = 4, **kwargs) -> list[FetchResult]:
    """Fetch several runs concurrently (one thread per *file*; layers within a file stay sequential)."""

    def one(req: FetchRequest) -> FetchResult:
        return AiwpSource(**kwargs).fetch_run(req)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, reqs))
