"""SoilGrids 2.0 at 1 km (12 properties x 5 depths) -> Zarr cube on a lat/lon grid.

Run from the repo root: uv run recipes/soilgrids.py
Individual steps:

    uv run recipes/soilgrids.py fetch         # download missing layers, hash all
    uv run recipes/soilgrids.py inspect       # check the source CRS, dtype, nodata
    uv run recipes/soilgrids.py build_zarr    # skipped if a finished store exists
    uv run recipes/soilgrids.py verify        # store vs source, exact, plus texture sum
    uv run recipes/soilgrids.py publish       # Hugging Face; never part of a plain run

The CHIRPS and CHIRTS cubes, with depth where they have time
------------------------------------------------------------
Each property is one ``(depth, lat, lon)`` int16 array, plain Zstandard, unsharded,
so GDAL 3.11+ reads it like the climate cubes. ``depth`` is the five standard
layers, indexed by their midpoint in cm, with ``depth_bnds`` holding the top and
bottom of each layer (CF bounds).

Reprojected, deliberately
-------------------------
ISRIC publishes the 1 km aggregates in Interrupted Goode Homolosine (metres). A
cube in metres pairs with no climate cube, so this one is warped onto a regular
1/120 deg EPSG:4326 grid (~926 m at the equator), west-aligned at -180 like the
CHIRPS/CHIRTS grids. Resampling is nearest-neighbour: every stored value is one
ISRIC published, never an interpolation of several.

Values are stored as published
------------------------------
SoilGrids maps properties as integers in "mapped units" (clay in g/kg, pH x 10,
...). Those integers are written unchanged; ``scale_factor`` is 1 / ISRIC's
conversion factor, so xarray decodes them to conventional units (clay in %, pH)
and the cube is lossless against the source. GDAL does not apply the scale.

Chunks are (5, 480, 480): one request returns a full profile over a 4-degree
tile, the expected read being point or bbox profiles of a few properties. 480
divides both grid axes exactly. Ocean chunks are never written.

Readers should only trust a store whose root attr ``build_complete`` is true.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BytesCodec

from cdh_data_pipeline import (
    GridSpec,
    cube_store,
    download,
    log,
    open_store,
    read_manifest,
    run,
    write_json,
)
from cdh_data_pipeline.daily_cube import store_stats
from cdh_data_pipeline.storage import clear_store
from cdh_data_pipeline.zarr import compressors

SOURCE = "https://files.isric.org/soilgrids/latest/data_aggregated/1000m"
# INPUT is the local GeoTIFF cache, named <var>_<depth>cm_mean.tif.
INPUT = os.environ.get("SOILGRIDS_INPUT", "input/soilgrids-1km")
# Local while testing; publish to s3://digital-atlas/cdh/data/soilgrids-v2-1km.
OUTPUT = os.environ.get("SOILGRIDS_OUTPUT", "output/soilgrids-v2-1km")
DATASET = "soilgrids-v2-1km"
STORE_NAME = f"{DATASET}.zarr"
URL = f"{OUTPUT}/{STORE_NAME}"
HF_REPO = os.environ.get("SOILGRIDS_HF_REPO", f"aaguilar90/{DATASET}")

#: Interrupted Goode Homolosine, stated explicitly: some GDAL builds mis-apply
#: ESRI:54052, which leaves gaps and shifted lobes in the warped grid.
SRC_CRS = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
RESAMPLING = "nearest"

#: Top-bottom in cm, in profile order. 100-200 cm is not in the 1 km aggregate set
#: downloaded here; adding it later means a rebuild, the depth axis is fixed.
DEPTHS = ("0-5", "5-15", "15-30", "30-60", "60-100")

#: 1/120 deg, 84 N to 60 S. The data spans about 56 S to 82.3 N; 60 S matches the
#: CHIRPS/CHIRTS south edge, and 144 deg of latitude is 36 whole 480-row chunks.
GRID = GridSpec(height=17280, width=43200, res=1 / 120, north=84.0, nodata=-32768)
FILL = -32768
CHUNKS = (len(DEPTHS), 480, 480)
#: What a partial store must match to be resumed rather than rebuilt.
LAYOUT = {"grid": [GRID.height, GRID.width, GRID.res, GRID.north, GRID.west],
          "chunks": list(CHUNKS), "depths": list(DEPTHS), "resampling": RESAMPLING}
# One thread per depth: each owns one source file, so no reader is shared.
WORKERS = len(DEPTHS)


@dataclass(frozen=True)
class SoilVariable:
    """One SoilGrids property. ``factor`` is ISRIC's mapped -> conventional divisor."""

    key: str
    long_name: str
    mapped_units: str
    factor: int
    units: str
    #: Plausible range in conventional units; verify reports values outside it.
    valid_min: float
    valid_max: float

    def zarr_attrs(self) -> dict:
        return {
            "long_name": self.long_name,
            "units": self.units,
            "scale_factor": 1 / self.factor,
            "add_offset": 0.0,
            "_FillValue": FILL,
            "mapped_units": self.mapped_units,
            "conversion_factor": self.factor,
            "grid_mapping": "spatial_ref",
            "coordinates": "depth_bnds spatial_ref",
        }


# Units and factors: https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html
VARIABLES = [
    SoilVariable("bdod", "Bulk density of the fine earth fraction",
                 "cg cm-3", 100, "kg dm-3", 0.0, 3.0),
    SoilVariable("cec", "Cation exchange capacity of the soil at pH 7",
                 "mmol(c) kg-1", 10, "cmol(c) kg-1", 0.0, 300.0),
    SoilVariable("cfvo", "Volumetric fraction of coarse fragments (> 2 mm)",
                 "cm3 dm-3", 10, "cm3 100cm-3", 0.0, 100.0),
    SoilVariable("clay", "Clay (< 0.002 mm) in the fine earth fraction",
                 "g kg-1", 10, "g 100g-1", 0.0, 100.0),
    SoilVariable("nitrogen", "Total nitrogen",
                 "cg kg-1", 100, "g kg-1", 0.0, 100.0),
    SoilVariable("phh2o", "Soil pH in water",
                 "pH x 10", 10, "pH", 0.0, 14.0),
    SoilVariable("sand", "Sand (> 0.05 mm) in the fine earth fraction",
                 "g kg-1", 10, "g 100g-1", 0.0, 100.0),
    SoilVariable("silt", "Silt (0.002-0.05 mm) in the fine earth fraction",
                 "g kg-1", 10, "g 100g-1", 0.0, 100.0),
    SoilVariable("soc", "Soil organic carbon in the fine earth fraction",
                 "dg kg-1", 10, "g kg-1", 0.0, 1000.0),
    SoilVariable("wv0010", "Volumetric water content at -10 kPa",
                 "10-3 cm3 cm-3", 10, "10-2 cm3 cm-3", 0.0, 100.0),
    SoilVariable("wv0033", "Volumetric water content at -33 kPa (field capacity)",
                 "10-3 cm3 cm-3", 10, "10-2 cm3 cm-3", 0.0, 100.0),
    SoilVariable("wv1500", "Volumetric water content at -1500 kPa (wilting point)",
                 "10-3 cm3 cm-3", 10, "10-2 cm3 cm-3", 0.0, 100.0),
]

ATTRS = {
    "title": "SoilGrids 2.0 global soil properties at 1 km",
    "institution": "ISRIC - World Soil Information",
    "source": (
        "SoilGrids 2.0 1000 m aggregated mean predictions "
        f"({SOURCE}/), reprojected from Interrupted Goode Homolosine"
    ),
    "references": "https://doi.org/10.5194/soil-7-217-2021",
    "license": "CC-BY-4.0",
    "id": DATASET,
}


def src_path(key, depth):
    return Path(INPUT, f"{key}_{depth}cm_mean.tif")


def depth_bounds():
    return np.array([[float(x) for x in d.split("-")] for d in DEPTHS])


# ------------------------------------------------------------------- source


def fetch():
    """Download any missing layer; hash and record every layer in the manifest.

    Files already in INPUT are adopted, not re-downloaded, so this is how a cache
    filled by another script gets its sources.json entries.
    """
    for var in VARIABLES:
        for d in DEPTHS:
            name = f"{var.key}_{d}cm_mean_1000.tif"
            download(f"{SOURCE}/{var.key}/{name}", src_path(var.key, d))


def _vrt(src):
    """``src`` warped onto GRID. Reads come back in the source's own integers."""
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    return WarpedVRT(
        src,
        src_crs=SRC_CRS,
        src_nodata=src.nodata if src.nodata is not None else FILL,
        crs=GRID.crs,
        transform=from_origin(GRID.west, GRID.north, GRID.res, GRID.res),
        width=GRID.width,
        height=GRID.height,
        resampling=Resampling[RESAMPLING],
        nodata=FILL,
        # GDAL's default approximation (0.125 px) interpolates across
        # Homolosine's lobe boundaries (the equator, the interruptions), which
        # misplaces cells there and depends on the window read. A tiny tolerance
        # makes it subdivide to exact; 0 itself trips a rasterio bug.
        tolerance=1e-6,
    )


def _read(vrt, row0, nrows, col0=0, ncols=None):
    """Window of the warped grid as int16 mapped units, nodata as FILL."""
    from rasterio.windows import Window

    ncols = GRID.width - col0 if ncols is None else ncols
    a = vrt.read(1, window=Window(col0, row0, ncols, nrows), masked=True)
    data = np.asarray(a.data, dtype=np.float32)
    missing = np.ma.getmaskarray(a) | ~np.isfinite(data)
    q = np.rint(np.where(missing, 0, data))
    if q.size and (q.min() < FILL + 1 or q.max() > np.iinfo(np.int16).max):
        # Mapped units are int16 at ISRIC; anything wider is a different product.
        raise ValueError(f"values {q.min()}..{q.max()} do not fit int16")
    out = q.astype(np.int16)
    out[missing] = FILL
    return out


def inspect():
    """Report what the source files are before anything is built from them.

    Stops on a missing layer or a non-Homolosine file: warping a raster from the
    wrong CRS succeeds silently and lands every value in the wrong place.
    """
    import rasterio

    missing = [
        str(src_path(v.key, d)) for v in VARIABLES for d in DEPTHS
        if not src_path(v.key, d).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} layer(s) missing, e.g. {missing[:3]}")
    shapes = set()
    for var in VARIABLES:
        for d in DEPTHS:
            with rasterio.open(src_path(var.key, d)) as src:
                shapes.add((src.width, src.height, src.dtypes[0], src.nodata,
                            tuple(round(x) for x in src.bounds)))
                crs = src.crs.to_proj4() if src.crs else "none"
                if src.crs and "igh" not in crs:
                    raise ValueError(f"{src.name}: CRS is {crs}, not Homolosine")
    log.info("%d layers, %d distinct (width, height, dtype, nodata, bounds):",
             len(VARIABLES) * len(DEPTHS), len(shapes))
    for s in sorted(shapes, key=str):
        log.info("  %s", s)
    # Latitude reach, measured on the warped grid itself: Homolosine bbox corners
    # sit outside the projection domain, so they cannot simply be transformed.
    # Data in GRID's first or last row means the source runs past it.
    step = 20
    with rasterio.open(src_path(VARIABLES[0].key, DEPTHS[0])) as src, _vrt(src) as vrt:
        coarse = vrt.read(1, out_shape=(GRID.height // step, GRID.width // step))
    rows = np.flatnonzero((coarse != FILL).any(axis=1))
    north = GRID.north - rows[0] * step * GRID.res
    south = GRID.north - (rows[-1] + 1) * step * GRID.res
    log.info("data spans lat %.2f..%.2f; GRID covers %.2f..%.2f",
             south, north, GRID.south, GRID.north)
    if rows[0] == 0 or rows[-1] == len(coarse) - 1:
        raise ValueError("source data reaches the edge of GRID; widen GRID")


# -------------------------------------------------------------------- build


def _built() -> bool:
    try:
        root = zarr.open_group(cube_store(URL), mode="r", use_consolidated=False)
    except Exception:  # noqa: BLE001  an unopenable store is an unbuilt one
        return False
    return bool(root.attrs.get("build_complete"))


def _write_coords(root):
    from rasterio.crs import CRS

    lat, lon = GRID.coords()
    for name, vals, unit, std, axis in (
        ("lat", lat, "degrees_north", "latitude", "Y"),
        ("lon", lon, "degrees_east", "longitude", "X"),
    ):
        a = root.create_array(name=name, shape=vals.shape, chunks=vals.shape,
                              dtype="float64", dimension_names=(name,))
        a[:] = vals
        a.attrs.update(units=unit, standard_name=std, axis=axis)

    bnds = depth_bounds()
    depth = root.create_array(name="depth", shape=(len(DEPTHS),), chunks=(len(DEPTHS),),
                              dtype="float64", dimension_names=("depth",))
    depth[:] = bnds.mean(axis=1)
    depth.attrs.update(units="cm", standard_name="depth", long_name="layer midpoint depth",
                       positive="down", axis="Z", bounds="depth_bnds")
    b = root.create_array(name="depth_bnds", shape=bnds.shape, chunks=bnds.shape,
                          dtype="float64", dimension_names=("depth", "bnds"))
    b[:] = bnds
    b.attrs.update(units="cm", long_name="layer top and bottom depth")

    wkt = CRS.from_string(GRID.crs).to_wkt()
    crs = root.create_array(name="spatial_ref", shape=(), chunks=(), dtype="int32")
    crs[...] = 0
    crs.attrs.update(grid_mapping_name="latitude_longitude", crs_wkt=wkt,
                     spatial_ref=wkt, semi_major_axis=6378137.0,
                     inverse_flattening=298.257223563)


def build_zarr():
    """Warp every layer onto GRID, one latitude band of all depths at a time.

    Peak RAM is about one (5, 480, 43200) band, ~200 MB of int16 plus float32
    scratch. Skipped when a finished store exists; delete it to rebuild.

    Resumable per variable: each array is marked ``complete`` once written, and
    a rerun over an unfinished store with the same layout skips those. A store
    with any other layout is cleared and started again.
    """
    import rasterio

    if _built():
        log.info("%s already built; delete it to rebuild", URL)
        return
    if not URL.rstrip("/").endswith(".zarr"):
        raise ValueError(f"refusing to overwrite non-.zarr store: {URL}")
    inspect()
    store = cube_store(URL)
    try:
        root = zarr.open_group(store, mode="r+", use_consolidated=False)
        resume = root.attrs.get("layout") == LAYOUT
    except Exception:  # noqa: BLE001  nothing usable there
        resume = False
    if not resume:
        clear_store(open_store(URL))
        root = zarr.create_group(store=store, overwrite=True)
        root.attrs.update(build_complete=False, layout=LAYOUT)
        _write_coords(root)

    rows = CHUNKS[1]
    bands = range(0, GRID.height, rows)
    with ThreadPoolExecutor(WORKERS) as pool:
        for k, var in enumerate(VARIABLES, 1):
            if var.key in root and root[var.key].attrs.get("complete"):
                log.info("[%d/%d] %s already written", k, len(VARIABLES), var.key)
                continue
            arr = root.create_array(
                overwrite=True,
                name=var.key,
                shape=(len(DEPTHS), GRID.height, GRID.width),
                chunks=CHUNKS,
                dtype="int16",
                fill_value=FILL,
                serializer=BytesCodec(),
                compressors=compressors("zstd"),
                dimension_names=("depth", "lat", "lon"),
            )
            arr.attrs.update(var.zarr_attrs())
            log.info("[%d/%d] %s", k, len(VARIABLES), var.key)
            with ExitStack() as stack:
                vrts = [
                    stack.enter_context(_vrt(stack.enter_context(
                        rasterio.open(src_path(var.key, d)))))
                    for d in DEPTHS
                ]
                for i, r0 in enumerate(bands, 1):
                    n = min(rows, GRID.height - r0)
                    arr[:, r0 : r0 + n, :] = np.stack(
                        list(pool.map(lambda v: _read(v, r0, n), vrts))
                    )
                    if i % 5 == 0 or i == len(bands):
                        log.info("  rows %d/%d", r0 + n, GRID.height)
            arr.attrs["complete"] = True

    g = GRID
    root.attrs.update({
        "Conventions": "CF-1.10",
        **ATTRS,
        "spatial_resolution_deg": g.res,
        "geospatial_lat_min": g.south,
        "geospatial_lat_max": g.north,
        "geospatial_lon_min": g.west,
        "geospatial_lon_max": g.east,
        "depth_layers_cm": list(DEPTHS),
        "variables": [v.key for v in VARIABLES],
        "source_crs": SRC_CRS,
        "resampling": RESAMPLING,
        "build_complete": True,
    })
    zarr.consolidate_metadata(store)
    total, n_files = store_stats(URL)
    log.info("built %s: %.1f MB / %d objects", URL, total / 1e6, n_files)


# ------------------------------------------------------------------- verify

#: (name, lat, lon) of 4-degree tiles checked by verify: land on every continent.
CHECKS = (("Kenya", 0.5, 37.5), ("Iowa", 42.0, -93.5), ("Punjab", 30.5, 75.5),
          ("Pampas", -34.5, -61.5), ("France", 47.0, 2.0), ("NSW", -31.0, 147.0))


def verify():
    """Re-warp sample tiles from the source and require exact equality.

    Exact, not approximate: the store holds the warped source integers
    unchanged, so any difference is a bug (wrong window, shifted grid, a write
    that never landed). Also reports decoded ranges, and clay + silt + sand,
    which ISRIC predicts separately and should still sum to about 100 %.
    """
    import rasterio
    import xarray as xr

    root = zarr.open_group(cube_store(URL), mode="r")
    if not root.attrs.get("build_complete"):
        raise RuntimeError(f"{URL} is not a finished build")
    lat, lon = GRID.coords()
    size = CHUNKS[1]
    bad = []
    for var in VARIABLES:
        arr = root[var.key]
        for d_i, d in enumerate(DEPTHS):
            with rasterio.open(src_path(var.key, d)) as src, _vrt(src) as vrt:
                for name, y, x in CHECKS:
                    r0 = int(np.abs(lat - y).argmin()) // size * size
                    c0 = int(np.abs(lon - x).argmin()) // size * size
                    ref = _read(vrt, r0, size, c0, size)
                    got = np.asarray(arr[d_i, r0 : r0 + size, c0 : c0 + size])
                    if not np.array_equal(ref, got):
                        bad.append(f"{var.key} {d}cm {name}: "
                                   f"{int((ref != got).sum())} cells differ")
    ds = xr.open_zarr(cube_store(URL), consolidated=True)
    for var in VARIABLES:
        top = ds[var.key].isel(depth=0)
        sample = top.isel(lat=slice(None, None, 37), lon=slice(None, None, 37)).values
        ok = sample[np.isfinite(sample)]
        if not ok.size:
            bad.append(f"{var.key}: no valid cells in the 0-5 cm sample")
            continue
        out = ((ok < var.valid_min) | (ok > var.valid_max)).mean()
        log.info("  %-8s %s: %.3g..%.3g, %.1f%% of sample valid%s", var.key,
                 var.units, ok.min(), ok.max(), 100 * ok.size / sample.size,
                 f", {100 * out:.3f}% outside plausible range" if out else "")
    if {"clay", "silt", "sand"} <= set(ds.data_vars):
        tex = sum(
            ds[k].isel(depth=0, lat=slice(None, None, 37), lon=slice(None, None, 37))
            for k in ("clay", "silt", "sand")
        ).values
        tex = tex[np.isfinite(tex)]
        log.info("  clay+silt+sand at 0-5 cm: median %.1f%%, 1st-99th pct "
                 "%.1f..%.1f%%", np.median(tex), *np.percentile(tex, [1, 99]))
    if bad:
        for b in bad[:20]:
            log.error("  %s", b)
        raise ValueError(f"{len(bad)} tile(s) differ from the source")
    log.info("all %d tiles match the source exactly",
             len(VARIABLES) * len(DEPTHS) * len(CHECKS))


# ---------------------------------------------------------- sidecars, publish


def write_sources():
    write_json(f"{OUTPUT}/sources.json", read_manifest(INPUT))


def _card() -> str:
    rows = "\n".join(
        f"| `{v.key}` | {v.long_name} | {v.units} | {v.mapped_units} | "
        f"{1 / v.factor:g} |"
        for v in VARIABLES
    )
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{STORE_NAME}"
    g = GRID
    return f"""---
pretty_name: {ATTRS["title"]}
license: cc-by-4.0
tags:
  - soil
  - soilgrids
  - soil properties
  - gridded
  - analysis-ready data
---

# {ATTRS["title"]}

SoilGrids 2.0 mean predictions of {len(VARIABLES)} soil properties at {len(DEPTHS)} \
standard depths, redistributed as a Zarr v3 datacube for bounding-box and \
profile queries. ISRIC's 1000 m aggregates are reprojected from Interrupted \
Goode Homolosine onto a regular {g.res:.7f} degree (1/120) EPSG:4326 grid with \
nearest-neighbour resampling, so every value is one ISRIC published. Values \
are stored as ISRIC's integer mapped units and decoded to conventional units \
with `scale_factor`; nothing is re-quantised.

## Coverage

| | |
|---|---|
| Bounding box | {g.west}, {g.south}, {g.east}, {g.north} (W, S, E, N) |
| Resolution | 1/120 degree (~926 m at the equator) |
| CRS | EPSG:4326 |
| Depths (cm) | {", ".join(DEPTHS)} |

## Variables

Each is an int16 array `(depth, lat, lon)`; nodata is {FILL}.

| Name | Property | Units (decoded) | Stored as | scale_factor |
|---|---|---|---|---|
{rows}

## Usage

```python
import xarray as xr

ds = xr.open_zarr("{url}", consolidated=True, chunks={{}})

# full profile at a point, all properties
profile = ds.sel(lat=0.5, lon=37.5, method="nearest").load()

# topsoil clay over a bounding box
clay = ds.clay.sel(depth=2.5, lat=slice(5.5, -4.7), lon=slice(33.9, 41.9)).load()
```

`depth` is indexed by layer midpoint (2.5, 10, 22.5, 45, 80 cm); `depth_bnds`
gives each layer's top and bottom. `lat` runs north to south.

## Reading from R or QGIS

Zarr v3 needs GDAL 3.11 or newer. Open a property by name:

```r
library(terra)
r <- rast('ZARR:"/path/to/{STORE_NAME}":/clay')
NAflag(r) <- {FILL}
r <- r / 10   # GDAL returns the stored integers; divide by the conversion factor
```

## Caveats

These are model predictions (quantile random forest), not measurements; ISRIC
publishes uncertainty layers separately, and they are not included here.
Nearest-neighbour reprojection duplicates or skips source cells where the
1 km equal-area grid and the 1/120 degree grid disagree in shape, chiefly at
high latitudes. Texture fractions are predicted independently, so clay + silt +
sand is close to but not exactly 100 %.

## Citation

> Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B.,
> Ribeiro, E., and Rossiter, D. (2021). SoilGrids 2.0: producing soil information
> for the globe with quantified spatial uncertainty. *SOIL*, 7, 217-240.
> https://doi.org/10.5194/soil-7-217-2021

SoilGrids is produced by ISRIC - World Soil Information and licensed CC BY 4.0.
"""


def write_card():
    """README.md for the Hub, beside the store."""
    open_store(OUTPUT).put("README.md", _card().encode())
    log.info("wrote %s/README.md", OUTPUT)


def publish():
    """Upload the store, README.md and sources.json to HF_REPO.

    Re-runnable: an interrupted upload resumes, and unchanged files are skipped.
    Needs HF_TOKEN or a cached `hf auth login`, and a local OUTPUT.
    """
    from huggingface_hub import HfApi

    if "://" in OUTPUT:
        raise ValueError("publish uploads from disk; OUTPUT must be a local path")
    if not _built():
        raise RuntimeError(f"{URL} is not a finished build")
    api = HfApi()
    log.info("authenticated as %s", api.whoami()["name"])
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO,
        repo_type="dataset",
        folder_path=URL,
        path_in_repo=STORE_NAME,
        commit_message=f"Upload {STORE_NAME}",
    )
    for name in ("README.md", "sources.json"):
        if Path(OUTPUT, name).exists():
            api.upload_file(path_or_fileobj=str(Path(OUTPUT, name)), path_in_repo=name,
                            repo_id=HF_REPO, repo_type="dataset",
                            commit_message=f"Update {name}")
    log.info("https://huggingface.co/datasets/%s", HF_REPO)


if __name__ == "__main__":
    steps = [fetch, inspect, build_zarr, verify, write_sources, write_card]
    # Publishing is outward-facing, so it only runs when named.
    if "publish" in sys.argv[1:]:
        steps.append(publish)
    run(*steps)
