"""GLW4 livestock rasters -> Zarr store and COGs.

Run from the repo root: uv run recipes/glw4.py
"""

import rioxarray  # noqa: F401  registers .rio
import xarray as xr

from cdh_data_pipeline import (
    blosc_zstd,
    open_raster,
    run,
    write_cog,
    write_zarr,
)

# INPUT/OUTPUT may be local paths or object-storage URLs.
INPUT = "https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA"
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


def load(code, name):
    url = f"{INPUT}/{SRC.format(code=code)}"
    da = open_raster(url, name)
    da.attrs.update(
        long_name=f"{name.capitalize()} density", units="head/km2", source_url=url
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
        url = f"{INPUT}/{SRC.format(code=code)}"
        write_cog(
            f"{OUTPUT}/cog/glw4-2020-{name}.tif",
            [url],
            [f"{name.capitalize()} density"],
            "head/km2",
        )


if __name__ == "__main__":
    run(build_zarr, build_cogs)
