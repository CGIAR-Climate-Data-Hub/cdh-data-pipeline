"""WorldPop age and sex structures (R2025A, 1 km, UN-adjusted) -> Zarr store.

Run from the repo root: uv run recipes/worldpop_age_sex.py

The source is one 15.6 GB zip per year (250 GB in all) and the server ignores
range requests. So each zip is downloaded, written into the store as one year,
then deleted. Local disk use peaks near 16 GB.

A rerun skips the years listed in INPUT/done.txt. Delete that file to rebuild
from scratch.
"""

from itertools import pairwise
from pathlib import Path

import rioxarray  # noqa: F401  registers .rio
import xarray as xr
from zarr.codecs import ZstdCodec
from zarr.storage import ObjectStore

from cdh_data_pipeline import download, log, open_raster, open_store, run, write_zarr

SOURCE = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A"
ZIP = "{year}/0_Mosaicked/v1/1km_ua/global_agesex_structures_{year}_CN_1km_R2025A_UA_v1.zip"
TIF = "global_{sex}_{age:02d}_{year}_CN_1km_R2025A_UA_v1.tif"
# INPUT holds the zip in flight and the resume log. Gitignored under input/.
INPUT = "input/worldpop-age-sex"
OUTPUT = "s3://digital-atlas/cdh/data/worldpop-age-sex-1km-ua-r2025a-v1"
ZARR = f"{OUTPUT}/worldpop-age-sex-1km-ua-r2025a-v1.zarr"

YEARS = list(range(2015, 2031))
# Lower bound of each age group in years: 0 (under 1), 1-4, 5-9, ..., 90+.
AGES = [0, 1, *range(5, 91, 5)]
AGE_LABELS = ["0", *(f"{a}-{b - 1}" for a, b in pairwise(AGES[1:])), "90+"]
# The zip's "t" files are f + m per age group, so they are left out.
SEXES = {"f": "female", "m": "male"}

# Expected read shape is some or all age/sex layers of one year over a country.
# One layer per chunk, so a read fetches only its layers. 540x540 cells is 4.5 degrees
# (~500 km at the equator). A layer splits into 4x10 shards of 8x8 chunks.
CHUNKS = (1, 1, 1, 540, 540)
SHARDS = (1, 1, 1, 4320, 4320)

ENCODING = {
    "population": {
        "chunks": CHUNKS,
        "shards": SHARDS,
        # Stored as int32 hundredths of a person: 2.5x smaller than float32.
        # Measured on one layer: the global sum shifts by 1e-5, a cell by 0.005.
        "dtype": "int32",
        "scale_factor": 0.01,
        "_FillValue": -1,
        # zarr's own fill: unwritten and all-nodata chunks then read as nodata.
        "fill_value": -1,
        # Plain zstd 9 beat blosc with any shuffle at a similar write speed.
        # Level 19 is 10% smaller but 20x slower to write.
        "compressors": (ZstdCodec(level=9),),
    }
}


def open_year(zip_path, year):
    """One year as a lazy population(year, age_group, sex, y, x) dataset."""

    def layer(age, sex):
        name = TIF.format(year=year, age=age, sex=sex)
        # GDAL reads the GeoTIFF inside the zip; one dask chunk per shard.
        chunks = {"y": SHARDS[-2], "x": SHARDS[-1]}
        return open_raster(f"/vsizip/{zip_path}/{name}", chunks=chunks)

    # join="override": take the first raster's grid, skip coordinate alignment.
    da = xr.concat(
        [
            xr.concat([layer(age, sex) for sex in SEXES], "sex", join="override")
            for age in AGES
        ],
        "age_group",
        join="override",
    )
    da = da.expand_dims(year=[year]).assign_coords(
        age_group=AGES,
        age_label=("age_group", AGE_LABELS),
        sex=list(SEXES),
        sex_name=("sex", list(SEXES.values())),
    )
    da.attrs.update(long_name="Population count", units="people per pixel")
    da.age_group.attrs.update(long_name="Lower bound of age group", units="years")
    ds = xr.Dataset({"population": da}).rio.write_crs("EPSG:4326")
    ds.attrs.update(
        title="WorldPop global age and sex structures 2015-2030, 1 km, UN-adjusted",
        source="WorldPop Global2 R2025A v1, constrained, UN-adjusted, 30 arc-second",
        references="https://hub.worldpop.org/geodata/listing?id=140",
    )
    return ds


def build_zarr():
    done_log = Path(INPUT) / "done.txt"
    done = done_log.read_text().split() if done_log.exists() else []
    for year in YEARS:
        if str(year) in done:
            continue
        zip_path = download(f"{SOURCE}/{ZIP.format(year=year)}", f"{INPUT}/{year}.zip")
        ds = open_year(zip_path, year)
        if not done:
            # Empty store covering every year. Each year then fills its own slot.
            empty = ds.reindex(year=YEARS)
            write_zarr(empty, ZARR, ENCODING, compute=False)
        # A region write only accepts variables on the region's dimension.
        ds = ds.drop_vars([c for c in ds.coords if c != "year"])
        ds.to_zarr(ObjectStore(open_store(ZARR)), region="auto")
        # Log only after the write, so a crash redoes the year.
        done.append(str(year))
        done_log.write_text("\n".join(done))
        zip_path.unlink()
        log.info("wrote %d (%d/%d)", year, len(done), len(YEARS))


if __name__ == "__main__":
    run(build_zarr)
