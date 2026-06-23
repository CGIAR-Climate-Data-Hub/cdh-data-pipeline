"""GLW4 livestock rasters -> efficient zarr store + cleanly-named COGs.

Run from the repo root: uv run recipes/glw4.py
"""

import xarray as xr

from cdh_data_pipeline import (
    blosc_zstd,
    open_raster,
    open_store,
    run,
    write_cogs,
    write_zarr,
)

# config: INPUT/OUTPUT are any local path or s3://, gs://, https:// URL
# (rasterio/GDAL auto-resolves https:// to /vsicurl, so sources stream over HTTP)
INPUT = "https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA"
OUTPUT = "s3://digital-atlas/cdh/data/glw4-2020"
SRC = "GLW4-2020.D-DA.{code}.tif"

# code -> readable name
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
    ds = xr.Dataset(das)
    ds.attrs.update(
        title="GLW4 2020 livestock density",
        source="Gridded Livestock of the World v4 (GLW4), 2020, dasymetric",
    )
    # ~1 MB spatial chunks (540px ~45deg) for regional reads; no sharding
    enc = {name: {"chunks": (540, 540), "compressors": (blosc_zstd(),)} for name in das}
    write_zarr(ds, f"{OUTPUT}/glw4-2020.zarr", enc)
    print(f"wrote {OUTPUT}/glw4-2020.zarr")


def build_cogs():
    cogs = open_store(f"{OUTPUT}/cog")
    jobs = [
        (
            f"glw4-2020-{name}.tif",
            f"{INPUT}/{SRC.format(code=code)}",
            f"{name.capitalize()} density",
            "head/km2",
        )
        for code, name in SPECIES.items()
    ]
    write_cogs(cogs, jobs)
    print(f"wrote {len(jobs)} COGs to {OUTPUT}/cog")


if __name__ == "__main__":
    run(build_zarr, build_cogs)
