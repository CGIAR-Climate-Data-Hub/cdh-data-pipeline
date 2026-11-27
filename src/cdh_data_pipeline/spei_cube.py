"""Inputs, tiles and the Zarr layout of the monthly SPEI cube.

The SPEI maths lives in :mod:`cdh_data_pipeline.spei` and knows nothing about
stores. This module reads the daily inputs lazily over HTTP, turns them into
monthly sums one spatial tile at a time, and writes the cube.

Inputs, all read through obstore (no full-store download):

* CHIRPS v3 daily precipitation and CHIRTS-ERA5 daily Tmax/Tmin: the Zarr v3
  cubes on Hugging Face, pinned to one repo revision for the whole build.
* AgERA5 FAO-56 Penman-Monteith ET0: the ECMWF ARCO Zarr v2 store
  (``geoChunked``, time-contiguous chunks), authenticated with the CDS key.

Every daily block is decoded, summed into its calendar month and dropped, so a
300 x 300 tile over 46 years needs a few hundred MB, never the daily stack.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import zarr
from obstore.store import HTTPStore
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.storage import ObjectStore

from cdh_data_pipeline import spei as S
from cdh_data_pipeline.daily_cube import GridSpec, cube_store
from cdh_data_pipeline.zarr import EPOCH, TIME_UNITS, open_for_write

HF = "https://huggingface.co"
AGERA5_ARCO = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-001/arco/"
    "sis_agrometeorological_indicators/all/geoChunked.zarr"
)
AGERA5_ET0 = "ReferenceET_PenmanMonteith_FAO56"
AGERA5_EPOCH = np.datetime64("1900-01-01", "D")

#: Tile edge in cells; matches the 300 x 300 chunks of the input cubes.
TILE = 300
#: Months per time chunk of the output arrays.
TIME_CHUNK = 120
SPEI_FILL = -32768
SPEI_SCALE = 0.01

WGS84_WKT = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",'
    '6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,'
    'ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic '
    'latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],'
    'AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",'
    '0.0174532925199433]],ID["EPSG",4326]]'
)


# ---------------------------------------------------------------- access


def hf_revision(repo_id: str) -> str:
    """Current commit SHA of a Hugging Face dataset repo (public API, no token)."""
    with urllib.request.urlopen(f"{HF}/api/datasets/{repo_id}", timeout=60) as r:
        return json.load(r)["sha"]


def hf_store_url(repo_id: str, revision: str, store_name: str) -> str:
    return f"{HF}/datasets/{repo_id}/resolve/{revision}/{store_name}"


def cds_key() -> str:
    """CDS API key from ``CDSAPI_KEY`` or ``~/.cdsapirc``."""
    if key := os.environ.get("CDSAPI_KEY"):
        return key
    rc = pathlib.Path(os.environ.get("CDSAPI_RC", pathlib.Path.home() / ".cdsapirc"))
    for line in rc.read_text().splitlines():
        if line.strip().startswith("key"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"no CDS key in CDSAPI_KEY or {rc}")


def open_http_group(url: str, headers: dict | None = None, zarr_format=None):
    opts = {"default_headers": headers} if headers else None
    store = HTTPStore.from_url(url, client_options=opts)
    return zarr.open_group(
        ObjectStore(store, read_only=True), mode="r", zarr_format=zarr_format
    )


def _bounded_map(fn, items, workers: int):
    """Like ``executor.map`` but with at most ``2 * workers`` results in memory."""
    items = list(items)
    with ThreadPoolExecutor(workers) as ex:
        pending = deque()
        for it in items:
            pending.append(ex.submit(fn, it))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


# ------------------------------------------------------------ the grid


@dataclass(frozen=True)
class Window:
    """A part of the target grid: rows/cols in the global CHIRPS grid."""

    row0: int
    row1: int
    col0: int
    col1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.row1 - self.row0, self.col1 - self.col0

    @property
    def key(self) -> str:
        return f"r{self.row0:04d}c{self.col0:04d}"


def subgrid(grid: GridSpec, bbox) -> tuple[GridSpec, Window]:
    """The cells of ``grid`` whose centres fall inside ``bbox`` (W, S, E, N)."""
    w, s, e, n = bbox
    lat, lon = grid.coords()
    rows = np.flatnonzero((lat > s) & (lat < n))
    cols = np.flatnonzero((lon > w) & (lon < e))
    if not len(rows) or not len(cols):
        raise ValueError(f"bbox {bbox} holds no cells of the grid")
    win = Window(int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
    sub = GridSpec(
        height=win.shape[0],
        width=win.shape[1],
        res=grid.res,
        north=round(grid.north - grid.res * win.row0, 6),
        west=round(grid.west + grid.res * win.col0, 6),
        nodata=grid.nodata,
    )
    return sub, win


def tiles(area: Window, size: int = TILE) -> list[Window]:
    """Tiles covering ``area``, cut on the global ``size`` grid.

    Cutting on global multiples of ``size`` keeps every tile inside one chunk
    column of the input cubes, whatever the pilot bbox.
    """
    out = []
    for r in range(area.row0 // size * size, area.row1, size):
        for c in range(area.col0 // size * size, area.col1, size):
            t = Window(
                max(r, area.row0),
                min(r + size, area.row1),
                max(c, area.col0),
                min(c + size, area.col1),
            )
            out.append(t)
    return out


# --------------------------------------------------------------- inputs


class DailyCube:
    """One int16 variable of a daily (time, lat, lon) cube on the CHIRPS-like grid."""

    def __init__(self, group, var: str, grid: GridSpec):
        self.array = group[var]
        attrs = dict(self.array.attrs)
        self.fill = attrs["_FillValue"]
        self.scale = float(attrs.get("scale_factor", 1.0))
        self.offset = float(attrs.get("add_offset", 0.0))
        self.dates = EPOCH + np.asarray(group["time"][:]).astype("timedelta64[D]")
        lat = np.asarray(group["lat"][:])
        lon = np.asarray(group["lon"][:])
        glat, glon = grid.coords()
        # Row/col of global target cell 0 in this cube; the cell centres must
        # coincide exactly (CHIRTS starts 200 rows further north than CHIRPS).
        self.row_shift = int(np.argmin(np.abs(lat - glat[0])))
        self.col_shift = int(np.argmin(np.abs(lon - glon[0])))
        if not (
            np.allclose(
                lat[self.row_shift : self.row_shift + len(glat)], glat, atol=1e-6
            )
            and np.allclose(
                lon[self.col_shift : self.col_shift + len(glon)], glon, atol=1e-6
            )
        ):
            raise ValueError(f"{var}: cell centres do not match the target grid")
        self.time_chunk = self.array.chunks[0]

    def read(self, t0: int, t1: int, win: Window) -> np.ndarray:
        """Days ``t0:t1`` over ``win``, physical float32, missing as NaN."""
        raw = self.array[
            t0:t1,
            win.row0 + self.row_shift : win.row1 + self.row_shift,
            win.col0 + self.col_shift : win.col1 + self.col_shift,
        ]
        out = raw.astype(np.float32) * np.float32(self.scale) + np.float32(self.offset)
        out[raw == self.fill] = np.nan
        return out

    def blocks(self, months) -> list[tuple[int, int]]:
        """Chunk-aligned day ranges covering ``months``."""
        mon = S.month_index(self.dates)
        idx = np.flatnonzero((mon >= months[0]) & (mon <= months[-1]))
        if not len(idx):
            return []
        c = self.time_chunk
        first = idx[0] // c * c
        return [
            (max(t, int(idx[0])), min(t + c, int(idx[-1]) + 1))
            for t in range(first, int(idx[-1]) + 1, c)
        ]


def monthly_from_daily(cube: DailyCube, months, win: Window, workers=8) -> np.ndarray:
    """Monthly sums of one daily cube over ``win``."""
    acc = S.MonthlyAccumulator(months, win.shape)

    def one(rng):
        t0, t1 = rng
        return cube.dates[t0:t1], cube.read(t0, t1, win)

    for dates, block in _bounded_map(one, cube.blocks(months), workers):
        acc.add(block, dates)
    return acc.result()


def monthly_hargreaves(
    tmax: DailyCube, tmin: DailyCube, months, win: Window, lat, workers=8
) -> tuple[np.ndarray, dict]:
    """Monthly Hargreaves-Samani PET over ``win`` and a Tmin > Tmax diagnostic.

    ``lat`` holds the latitude of each row of ``win``.
    """
    if not np.array_equal(tmax.dates, tmin.dates):
        raise ValueError("Tmax and Tmin have different time axes")
    acc = S.MonthlyAccumulator(months, win.shape)
    inverted = np.zeros(win.shape, np.int32)  # days with Tmin > Tmax, per cell
    inverted_by_year: dict[int, int] = {}
    max_excess = np.zeros(win.shape, np.float32)  # largest Tmin - Tmax, degC

    def one(rng):
        t0, t1 = rng
        return tmax.dates[t0:t1], tmax.read(t0, t1, win), tmin.read(t0, t1, win)

    for dates, tx, tn in _bounded_map(one, tmax.blocks(months), workers):
        doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
        ra = S.ra_mm_day(doy, lat)[:, :, None]
        acc.add(S.hargreaves_et0(tx, tn, ra), dates)
        with np.errstate(invalid="ignore"):
            inv = tn > tx
        inverted += inv.sum(0)
        excess = np.where(inv, tn - tx, 0).max(0)
        np.maximum(max_excess, excess, out=max_excess)
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        for y in np.unique(years):
            inverted_by_year[int(y)] = inverted_by_year.get(int(y), 0) + int(
                inv[years == y].sum()
            )
    diag = {
        "tmin_gt_tmax_days": inverted,
        "tmin_gt_tmax_max_excess": max_excess,
        "by_year": inverted_by_year,
    }
    return acc.result(), diag


class Agera5ET0:
    """AgERA5 FAO-56 Penman-Monteith ET0 from the ECMWF ARCO store."""

    def __init__(self, url: str = AGERA5_ARCO, headers: dict | None = None):
        g = open_http_group(url, headers, zarr_format=2)
        self.array = g[AGERA5_ET0]
        self.fill = self.array.attrs.get("missing_value", -9999.0)
        self.dates = AGERA5_EPOCH + np.asarray(g["time"][:]).astype("timedelta64[D]")
        self.lat = np.round(np.asarray(g["latitude"][:]), 6)
        self.lon = np.round(np.asarray(g["longitude"][:]), 6)
        self.time_chunk = self.array.chunks[0]
        self.attrs = dict(g.attrs)

    def monthly_on(self, months, lat, lon, workers=4) -> np.ndarray:
        """Monthly ET0 sums regridded bilinearly to cell centres ``lat`` x ``lon``.

        Sums are made on the 0.1 degree points first, then regridded: the
        bilinear weights are fixed, so this equals regridding every day, at a
        quarter of the cost.
        """
        ylat = S.axis_weights(self.lat, lat)
        xlon = S.axis_weights(self.lon, lon, periodic=360.0)
        rows = np.unique(np.concatenate([ylat.lo, ylat.hi]))
        cols = np.unique(np.concatenate([xlon.lo, xlon.hi]))
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        # Weights re-indexed into the block that is actually read.
        ylat = S.AxisWeights(ylat.lo - r0, ylat.hi - r0, ylat.w_hi)
        cpos = {c: i for i, c in enumerate(cols)}
        xlon = S.AxisWeights(
            np.array([cpos[c] for c in xlon.lo]),
            np.array([cpos[c] for c in xlon.hi]),
            xlon.w_hi,
        )
        acc = S.MonthlyAccumulator(months, (r1 - r0, len(cols)))
        mon = S.month_index(self.dates)
        idx = np.flatnonzero((mon >= months[0]) & (mon <= months[-1]))
        c = self.time_chunk
        ranges = [
            (max(t, int(idx[0])), min(t + c, int(idx[-1]) + 1))
            for t in range(idx[0] // c * c, int(idx[-1]) + 1, c)
        ]

        def one(rng):
            t0, t1 = rng
            raw = self.array.get_orthogonal_selection(
                (slice(t0, t1), slice(r0, r1), cols)
            )
            raw = np.asarray(raw, np.float32)
            raw[(raw == self.fill) | ~np.isfinite(raw)] = np.nan
            return self.dates[t0:t1], raw

        for dates, block in _bounded_map(one, ranges, workers):
            acc.add(block, dates)
        return S.regrid_bilinear(acc.result(), ylat, xlon)


# ------------------------------------------------------------ the store


def cube_store_url(url: str):
    """Zarr store handle for a local path or object-store url (as the daily cubes)."""
    return cube_store(url)


def open_store_rw(url: str):
    """The cube for writing, with metadata read fresh (see ``open_for_write``)."""
    return open_for_write(cube_store(url))


def consolidate(url: str) -> None:
    zarr.consolidate_metadata(cube_store(url))


def month_days(months) -> np.ndarray:
    """First day of each month as int32 days since the cube epoch."""
    return (np.asarray(months, "datetime64[M]").astype("datetime64[D]") - EPOCH).astype(
        "int32"
    )


def bounds_days(months, window: int) -> np.ndarray:
    return (S.window_bounds(months, window) - EPOCH).astype("int32")


def spei_name(window: int) -> str:
    return f"spei_{window:02d}"


def _chunks(grid: GridSpec) -> tuple[int, int]:
    return min(TILE, grid.height), min(TILE, grid.width)


def create_store(
    url_or_store, grid: GridSpec, months, windows, root_attrs: dict, var_attrs: dict
):
    """Create the empty cube: coordinates, bounds and every data array.

    ``var_attrs`` maps array name -> extra attributes (method descriptions).
    Arrays already present are left alone, so adding a window later only adds
    its arrays.
    """
    root = zarr.open_group(url_or_store, mode="a")
    root.attrs.update(root_attrs)
    n = len(months)
    lat, lon = grid.coords()
    cy, cx = _chunks(grid)
    zstd = [ZstdCodec(level=5)]

    def array(name, shape, chunks, dtype, dims, fill, attrs):
        if name in root:
            return root[name]
        a = root.create_array(
            name=name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            dimension_names=dims,
            fill_value=fill,
            serializer=BytesCodec(),
            compressors=zstd,
        )
        a.attrs.update(attrs)
        return a

    t = array(
        "time",
        (n,),
        (max(n, TIME_CHUNK),),
        "int32",
        ("time",),
        0,
        {
            "units": TIME_UNITS,
            "calendar": "proleptic_gregorian",
            "standard_name": "time",
            "axis": "T",
            "long_name": "first day of the (last) month",
            "bounds": "time_bnds",
        },
    )
    t[:] = month_days(months)
    tb = array(
        "time_bnds",
        (n, 2),
        (max(n, TIME_CHUNK), 2),
        "int32",
        ("time", "bnds"),
        0,
        {
            "units": TIME_UNITS,
            "calendar": "proleptic_gregorian",
            "long_name": "calendar month bounds of precip and pet",
        },
    )
    tb[:] = bounds_days(months, 1)
    for nm, vals, unit, std, axis in (
        ("lat", lat, "degrees_north", "latitude", "Y"),
        ("lon", lon, "degrees_east", "longitude", "X"),
    ):
        a = array(
            nm,
            vals.shape,
            vals.shape,
            "float64",
            (nm,),
            0.0,
            {
                "units": unit,
                "standard_name": std,
                "axis": axis,
                "long_name": std,
            },
        )
        a[:] = vals
    if "spatial_ref" not in root:
        crs = root.create_array(
            name="spatial_ref", shape=(), dtype="int32", fill_value=0
        )
        crs[...] = 0
        crs.attrs.update(
            {
                "grid_mapping_name": "latitude_longitude",
                "crs_wkt": WGS84_WKT,
                "spatial_ref": WGS84_WKT,
                "epsg_code": "EPSG:4326",
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            }
        )
    m = array(
        "month",
        (12,),
        (12,),
        "int8",
        ("month",),
        0,
        {
            "long_name": "calendar month of the fitted distribution (1 = January)",
        },
    )
    m[:] = np.arange(1, 13, dtype="int8")

    space = (grid.height, grid.width)
    data_dims = ("time", "lat", "lon")
    for nm in ("precip", "pet"):
        array(
            nm,
            (n, *space),
            (TIME_CHUNK, cy, cx),
            "float32",
            data_dims,
            np.nan,
            {
                "grid_mapping": "spatial_ref",
                **var_attrs.get(nm, {}),
            },
        )
    for w in windows:
        name = spei_name(w)
        b = array(
            f"time_bnds_{w:02d}",
            (n, 2),
            (max(n, TIME_CHUNK), 2),
            "int32",
            ("time", "bnds"),
            0,
            {
                "units": TIME_UNITS,
                "calendar": "proleptic_gregorian",
                "long_name": f"bounds of the {w}-month window of {name}",
            },
        )
        b[:] = bounds_days(months, w)
        array(
            name,
            (n, *space),
            (TIME_CHUNK, cy, cx),
            "int16",
            data_dims,
            SPEI_FILL,
            {
                "scale_factor": SPEI_SCALE,
                "add_offset": 0.0,
                "_FillValue": SPEI_FILL,
                "units": "1",
                "grid_mapping": "spatial_ref",
                "window_months": w,
                "window_bounds": f"time_bnds_{w:02d}",
                "valid_min": -300,
                "valid_max": 300,
                **var_attrs.get(name, {}),
            },
        )
        for p, desc in (
            ("xi", "location"),
            ("alpha", "scale"),
            ("k", "shape"),
        ):
            array(
                f"{name}_{p}",
                (12, *space),
                (12, cy, cx),
                "float32",
                ("month", "lat", "lon"),
                np.nan,
                {
                    "long_name": f"generalized logistic {desc} parameter ({p}) "
                    f"of the {w}-month water balance, per calendar month",
                    "units": "mm" if p != "k" else "1",
                    "grid_mapping": "spatial_ref",
                },
            )
    return root


def extend_time(root, months, windows) -> None:
    """Grow every time-dimensioned array to ``months`` (the new full axis)."""
    n = len(months)
    for name in list(root.array_keys()):
        a = root[name]
        dims = a.metadata.dimension_names
        if dims and dims[0] == "time" and a.shape[0] < n:
            a.resize((n, *a.shape[1:]))
    root["time"][:] = month_days(months)
    root["time_bnds"][:] = bounds_days(months, 1)
    for w in windows:
        root[f"time_bnds_{w:02d}"][:] = bounds_days(months, w)


def encode_spei(v: np.ndarray) -> np.ndarray:
    out = np.where(np.isfinite(v), np.rint(v / SPEI_SCALE), SPEI_FILL)
    return out.astype("int16")


def decode_spei(a: np.ndarray) -> np.ndarray:
    out = a.astype("f4") * np.float32(SPEI_SCALE)
    out[a == SPEI_FILL] = np.nan
    return out


# -------------------------------------------------------- tile progress


class TileLog:
    """Which tiles each step has finished, kept beside the store as JSON.

    Saved after every tile (write to a temporary file, then rename), so a stop
    loses at most the tile in progress.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = pathlib.Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def done(self, step: str) -> dict:
        return self.data.setdefault(step, {})

    def mark(self, step: str, key: str, info: dict) -> None:
        self.done(step)[key] = info
        self.save()

    def reset(self, step: str) -> None:
        self.data[step] = {}
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)
