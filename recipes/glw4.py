"""GLW4 livestock rasters -> Zarr store and COGs.

Run from the repo root: uv run recipes/glw4.py
"""

import rioxarray  # noqa: F401  registers .rio
import xarray as xr

from cdh_data_pipeline import (
    blosc_zstd,
    download,
    open_raster,
    read_manifest,
    run,
    write_cog,
    write_json,
    write_zarr,
)

SOURCE = "https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA"
# INPUT is the local GeoTIFF cache. Gitignored under input/.
INPUT = "input/glw4"
OUTPUT = "s3://digital-atlas/cdh/data/glw4-2020"
SRC = "GLW4-2020.D-DA.{code}.tif"

SPECIES = {
    "BFL": "buffalo",
    "CHK": "chicken",
    "CTL": "cattle",
    "GTS": "goat",
    "PGS": "pig",
    "SHP": "sheep",
}


def fetch():
    """Download the source GeoTIFFs into INPUT (skips any already present)."""
    for code in SPECIES:
        name = SRC.format(code=code)
        download(f"{SOURCE}/{name}", f"{INPUT}/{name}")


def load(code, name):
    src = SRC.format(code=code)
    da = open_raster(f"{INPUT}/{src}", name)
    da.attrs.update(
        long_name=f"{name.capitalize()} density",
        units="head/km2",
        source_url=f"{SOURCE}/{src}",
    )
    return da


def build_zarr():
    das = {name: load(code, name) for code, name in SPECIES.items()}
    ds = xr.Dataset(das).rio.write_crs("EPSG:4326")
    ds.attrs.update(
        title="GLW4 2020 livestock density",
        source="Gridded Livestock of the World v4 (GLW4), 2020, dasymetric",
    )
    # write_multiscale_zarr(..., layout="level") for overview pyramids
    enc = {
        v: {"chunks": (1080, 1080), "compressors": (blosc_zstd(),)}
        for v in ds.data_vars
    }
    write_zarr(ds, f"{OUTPUT}/glw4-2020.zarr", enc)


def build_cogs():
    for code, name in SPECIES.items():
        write_cog(
            f"{OUTPUT}/cog/glw4-2020-{name}.tif",
            [f"{INPUT}/{SRC.format(code=code)}"],
            [f"{name.capitalize()} density"],
            "head/km2",
        )


def write_metadata():
    write_json(f"{OUTPUT}/sources.json", read_manifest(INPUT))


if __name__ == "__main__":
    run(fetch, build_zarr, build_cogs, write_metadata)
