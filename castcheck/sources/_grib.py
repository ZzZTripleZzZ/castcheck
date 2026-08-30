"""Decode a single GRIB message (fetched as a byte range) into station values.

Shared by `ecmwf.py` and `gfs.py`. `cfgrib` needs a real file, so the bytes are written to a temporary
file that is deleted immediately after the values are read. Set ``CASTCHECK_TMPDIR`` to control where
those files land (default: the system temp directory).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..config import Station
from ..grid import extract_all, is_fill
from .base import k_to_c

log = logging.getLogger(__name__)

READ_KEYS = ["generatingProcessIdentifier", "tablesVersion"]


@dataclass
class DecodedField:
    """Station values (°C) plus the GRIB keys we use for `model_version`."""

    values: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    n_fill: int = 0
    generating_process: int | None = None
    error: str = ""


def tmpdir() -> str | None:
    return os.environ.get("CASTCHECK_TMPDIR") or None


def decode_message(raw: bytes, stations: list[Station]) -> DecodedField:
    """Extract bilinear/nearest station values in °C from one GRIB message.

    Values that fail the Kelvin sanity check (`grid.is_fill`) come back as ``None`` so the caller can
    write a ``fill_value`` missing row.
    """
    if not raw:
        return DecodedField(error="empty_body")
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".grib2", prefix="castcheck-", dir=tmpdir())
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        with xr.open_dataset(
            path, engine="cfgrib", backend_kwargs={"indexpath": "", "read_keys": READ_KEYS}
        ) as ds:
            names = list(ds.data_vars)
            if not names:
                return DecodedField(error="no_field")
            da = ds[names[0]]
            gpid = da.attrs.get("GRIB_generatingProcessIdentifier")
            arr = np.asarray(da.values, dtype=float)
            if arr.ndim != 2:
                arr = arr.reshape(arr.shape[-2], arr.shape[-1])
            lats = np.asarray(ds["latitude"].values, dtype=float)
            lons = np.asarray(ds["longitude"].values, dtype=float)
            raw_vals = extract_all(arr, lats, lons, stations)
    except Exception as exc:  # noqa: BLE001 - cfgrib/eccodes raise many types; a bad message must never abort a run
        log.warning("GRIB decode failed: %s", exc)
        return DecodedField(error="decode_error")
    finally:
        if path and os.path.exists(path):
            os.unlink(path)

    out: dict[str, tuple[float | None, float | None]] = {}
    n_fill = 0
    for sid, (bl, nn) in raw_vals.items():
        pair: list[float | None] = []
        for v in (bl, nn):
            if is_fill(v):
                pair.append(None)
                n_fill += 1
            else:
                pair.append(k_to_c(v))
        out[sid] = (pair[0], pair[1])
    return DecodedField(values=out, n_fill=n_fill, generating_process=int(gpid) if gpid is not None else None)
