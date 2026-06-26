"""GLW4 livestock rasters -> efficient zarr store + cleanly-named COGs.

Run from the repo root: uv run recipes/glw4.py
"""

import rioxarray  # noqa: F401  registers the .rio accessor used by write_crs
import xarray as xr

from cdh_data_pipeline import (
    make_cog,
    open_raster,
    open_store,
    run,
    write_multiscale_zarr,
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
    ds = xr.Dataset(das).rio.write_crs("EPSG:4326")
    ds.attrs.update(
        title="GLW4 2020 livestock density",
        source="Gridded Livestock of the World v4 (GLW4), 2020, dasymetric",
    )
    # one multiscale GeoZarr store: /<species>/0 = native + /1../3 overviews (x2 each),
    # one multiscales group per species. Serves regional analysis, deck.gl-raster, and
    # GDAL from one store. All species are densities (head/km2) -> mean.
    write_multiscale_zarr(ds, f"{OUTPUT}/glw4-2020.zarr", factors=[2, 4, 8])
    print(f"wrote {OUTPUT}/glw4-2020.zarr")


def build_cogs():
    cogs = open_store(f"{OUTPUT}/cog")
    for code, name in SPECIES.items():
        url = f"{INPUT}/{SRC.format(code=code)}"
        cogs.put(
            f"glw4-2020-{name}.tif",
            make_cog([url], [f"{name.capitalize()} density"], "head/km2"),
        )
    print(f"wrote {len(SPECIES)} COGs to {OUTPUT}/cog")


if __name__ == "__main__":
    run(build_zarr, build_cogs)
