"""Small shared helpers for dated-raster archives. Nothing here knows a mission."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

#: One row of an Apache/nginx autoindex table: filename, size, modification date.
AUTOINDEX_ROW = re.compile(
    r'<a href="(?P<name>[^"?/][^"]*\.(?:tif|tiff|nc))"[^>]*>.*?'
    r'<td class="size">\s*(?P<size>[^<]+?)\s*</td>.*?'
    r'<td class="date">\s*(?P<date>[^<]+?)\s*</td>',
    re.IGNORECASE | re.DOTALL,
)

#: ``chirps-v3.0.rnl.2026.08.31.tif`` / ``CHIRTS-ERA5.daily_Tmax.2001.01.01.tif``
DOTTED_DATE = re.compile(r"\.(\d{4})\.(\d{2})\.(\d{2})\.(?:tif|tiff|nc)$", re.IGNORECASE)
#: ``..._AgERA5_20010101_final-v2.0.0.nc``
COMPACT_DATE = re.compile(r"[_.](\d{4})(\d{2})(\d{2})[_.]")

_SIZE = re.compile(r"^([\d.]+)\s*([KMGT]?)i?B$", re.IGNORECASE)
_UNIT = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

#: Autoindex sizes are rounded to one decimal of a binary unit, so a parsed size
#: is accurate only to ~0.05 of that unit -- fine for change detection, useless
#: for verifying a transfer (use Content-Length for that).
SIZE_TOLERANCE = 0.02


@dataclass(frozen=True)
class MeteoVariable:
    """One daily meteorological variable and how it is stored.

    ``scale`` and ``offset`` define the stored representation:
    ``stored = round((physical - offset) / scale)``, inverted on read. The integer
    range is chosen so the observed physical range fits with headroom, and ``fill``
    is reserved outside it for missing data.

    Attributes
    ----------
    key
        Short handle used on the command line and in state (``tmax``).
    nc_name
        Array name in the cube, and the variable name in a source NetCDF.
    folder
        Directory holding this variable in a local archive.
    """

    key: str
    nc_name: str
    folder: str
    dtype: str
    scale: float
    offset: float
    fill: int
    units: str
    long_name: str
    standard_name: str
    #: Physical plausibility bounds, checked on ingest.
    valid_min: float
    valid_max: float

    @property
    def quantum(self) -> float:
        """Smallest representable step, in physical units."""
        return self.scale

    @property
    def store_bounds(self) -> tuple[int, int]:
        """Integer range usable for data, with ``fill`` excluded."""
        info = np.iinfo(self.dtype)
        lo, hi = int(info.min), int(info.max)
        # Reserve only the sentinel itself; clipping one extra step off the bottom
        # would silently destroy genuine zeros -- solar radiation has ~157k true
        # zero pixels per day during polar night.
        if self.fill <= lo:
            lo += 1
        elif self.fill >= hi:
            hi -= 1
        return lo, hi

    def encode(self, arr: np.ndarray) -> np.ndarray:
        """Physical float values -> stored integers. Non-finite becomes fill."""
        missing = ~np.isfinite(arr)
        q = (arr.astype(np.float64) - self.offset) / self.scale
        np.rint(q, out=q)
        lo, hi = self.store_bounds
        # Zero the holes before clipping/casting: casting NaN to an integer is
        # undefined behaviour in numpy and raises a RuntimeWarning.
        q[missing] = 0
        np.clip(q, lo, hi, out=q)
        out = q.astype(self.dtype)
        out[missing] = self.fill
        return out

    def decode(self, arr: np.ndarray) -> np.ndarray:
        """Stored integers -> physical float32, with fill as NaN."""
        out = arr.astype(np.float32) * np.float32(self.scale) + np.float32(self.offset)
        out[arr == self.fill] = np.nan
        return out

    def check_range(self, arr: np.ndarray) -> list[str]:
        """Flag physically implausible values before they enter the cube."""
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return ["all values are missing"]
        problems = []
        if finite.min() < self.valid_min:
            problems.append(
                f"minimum {finite.min():.3f} below valid_min "
                f"{self.valid_min} {self.units}"
            )
        if finite.max() > self.valid_max:
            problems.append(
                f"maximum {finite.max():.3f} above valid_max "
                f"{self.valid_max} {self.units}"
            )
        return problems

    def zarr_attrs(self) -> dict:
        return {
            "long_name": self.long_name,
            "standard_name": self.standard_name,
            "units": self.units,
            "scale_factor": self.scale,
            "add_offset": self.offset,
            "_FillValue": self.fill,
            "grid_mapping": "spatial_ref",
        }


def parse_listing_size(text: str) -> int | None:
    m = _SIZE.match(text.strip())
    return int(float(m.group(1)) * _UNIT[m.group(2).upper()]) if m else None


def parse_dotted_date(name: str | Path) -> date:
    """Pull a date out of a filename using either supported convention."""
    stem = Path(name).name
    for pattern in (DOTTED_DATE, COMPACT_DATE):
        m = pattern.search(stem)
        if m:
            return date(*map(int, m.groups()))
    raise ValueError(f"no date in {stem!r}")


def dense_daily_axis(first: date, last: date) -> np.ndarray:
    """Every calendar day from ``first`` to ``last`` inclusive.

    Cubes always carry a dense axis: a gap in the source becomes a nodata day
    rather than a missing row, because compressing the axis would silently shift
    every date lookup after the gap.
    """
    return np.arange(
        np.datetime64(first, "D"),
        np.datetime64(last, "D") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )


def jsonify(obj: Any) -> Any:
    """Make numpy / Path / date values JSON-serialisable for Zarr attributes."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj
