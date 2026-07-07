"""MapSPAM 2020 V2r2 rasters -> Zarr cube and per-technology COGs.

Run from the repo root with DATAVERSE_TOKEN set:
uv run --env-file .env recipes/mapspam.py
"""

import dask
import xarray as xr

from cdh_data_pipeline import (
    blosc_zstd,
    download_dataverse,
    open_raster,
    run,
    write_cog,
    write_json,
    write_zarr,
)

# INPUT is the local Dataverse zip cache. OUTPUT may be local or object storage.
INPUT = "input/spam_ifpri"
OUTPUT = "s3://digital-atlas/cdh/data/mapspam2020-v2r2"
WORKERS = 4

DOI = "doi:10.7910/DVN/SWPENT"
VERSION = "6.0"

# MapSPAM code -> (variable name, units)
VARS = {
    "A": ("physical_area", "ha"),
    "H": ("harvested_area", "ha"),
    "P": ("production", "mt"),
    "Y": ("yield", "kg/ha"),
}
TECH = {"A": "all", "I": "irrigated", "R": "rainfed"}

CROPS = {
    "whea": "Wheat",
    "rice": "Rice",
    "maiz": "Maize",
    "barl": "Barley",
    "mill": "Small Millet",
    "pmil": "Pearl Millet",
    "sorg": "Sorghum",
    "ocer": "Other Cereals",
    "pota": "Potato",
    "swpo": "Sweet Potato",
    "yams": "Yams",
    "cass": "Cassava",
    "orts": "Other Roots",
    "bean": "Bean",
    "chic": "Chickpea",
    "cowp": "Cowpea",
    "pige": "Pigeon Pea",
    "lent": "Lentil",
    "opul": "Other Pulses",
    "soyb": "Soybean",
    "grou": "Groundnut",
    "cnut": "Coconut",
    "oilp": "Oilpalm",
    "sunf": "Sunflower",
    "rape": "Rapeseed",
    "sesa": "Sesame Seed",
    "ooil": "Other Oil Crops",
    "sugc": "Sugarcane",
    "sugb": "Sugarbeet",
    "cott": "Cotton",
    "ofib": "Other Fibre Crops",
    "coff": "Arabic Coffee",
    "rcof": "Robusta Coffee",
    "coco": "Cocoa",
    "teas": "Tea",
    "toba": "Tobacco",
    "bana": "Banana",
    "plnt": "Plantain",
    "citr": "Citrus",
    "trof": "Other Tropical Fruit",
    "temf": "Temperate Fruit",
    "toma": "Tomato",
    "onio": "Onion",
    "vege": "Other Vegetables",
    "rubb": "Rubber",
    "rest": "Rest of Crops",
}


def src(vcode, name, crop, tech):
    """Return the GDAL /vsizip path for one source layer."""
    zip_ = f"{INPUT}/spam2020V2r2_global_{name}.geotiff.zip"
    tif = f"spam2020V2r2_global_{name}/spam2020_V2r2_global_{vcode}_{crop.upper()}_{tech}.tif"
    return f"/vsizip/{zip_}/{tif}"


def build_var(vcode, name, units):
    """Build one variable with dimensions (technology, crop, y, x)."""
    # Source rasters share a grid but differ by small coordinate noise.
    da = xr.concat(
        [
            xr.concat(
                [open_raster(src(vcode, name, c, t), chunks=-1) for c in CROPS],
                dim="crop",
                join="override",
            )
            for t in TECH
        ],
        dim="technology",
        join="override",
    )
    da = da.assign_coords(technology=list(TECH.values()), crop=list(CROPS))
    da.attrs.update(long_name=name.replace("_", " ").capitalize(), units=units)
    return da.rename(name)


def fetch():
    """Download the source GeoTIFF zips into INPUT (skips any already present)."""
    expected = [f"spam2020V2r2_global_{name}.geotiff.zip" for name, _ in VARS.values()]
    download_dataverse(DOI, expected, INPUT, version=VERSION)


def build_zarr():
    dask.config.set(scheduler="threads", num_workers=WORKERS)
    das = {name: build_var(c, name, u) for c, (name, u) in VARS.items()}
    ref = next(iter(das.values()))
    grid = {"y": ref.y.values, "x": ref.x.values}
    das = {name: da.assign_coords(grid) for name, da in das.items()}
    ds = xr.Dataset(das)
    ds = ds.assign_coords(crop_name=("crop", list(CROPS.values())))
    ds.attrs.update(
        title="MapSPAM 2020 V2r2 - spatially-disaggregated crop statistics",
        institution="International Food Policy Research Institute (IFPRI)",
        source="Global SPAM 2020 V2r2 (5 arc-minute, dasymetric)",
        references="https://doi.org/10.7910/DVN/SWPENT",
    )

    # Expected read shape is one crop and one technology over a bbox. 90x90 cells
    # is 7.5 degrees (~830 km at the equator). Each shard holds one full layer.
    encoding = {
        v: {
            "chunks": (1, 1, 90, 90),
            "shards": (1, 1, ds.sizes["y"], ds.sizes["x"]),
            "compressors": (blosc_zstd(),),
        }
        for v in ds.data_vars
    }
    write_zarr(ds, f"{OUTPUT}/spam2020-v2r2.zarr", encoding)


def build_cogs():
    """Write one 46-band COG per variable and technology.

    Running sequentially for now as each COG peaks around 2.5 GB.
    """
    for vcode, (name, units) in VARS.items():
        for tcode, tname in TECH.items():
            srcs = [src(vcode, name, crop, tcode) for crop in CROPS]
            out = f"{OUTPUT}/cog/spam2020-{name}-{tname}.tif"
            # Band names are the SPAM crop codes (matches the zarr crop coord);
            # clean names ride along as per-band long_name tags.
            write_cog(
                out,
                srcs,
                list(CROPS),
                units,
                long_names=list(CROPS.values()),
                interleave="BAND",
            )


def write_metadata():
    write_json(f"{OUTPUT}/crop-codes.json", CROPS)


if __name__ == "__main__":
    run(fetch, build_zarr, build_cogs, write_metadata)
