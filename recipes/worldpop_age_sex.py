"""WorldPop age and sex structures (R2025A, 1 km, UN-adjusted) -> Zarr store.

Run from the repo root: uv run recipes/worldpop_age_sex.py
Then, once the local store looks right:
    uv run --env-file .env recipes/worldpop_age_sex.py publish

The source is one 15.6 GB zip per year (250 GB in all). The server ignores
range requests and caps each connection at ~5-9 MB/s, so fetch downloads
PREFETCH zips at once. Each zip's 40 female and male GeoTIFFs are extracted
(11 GB a year, 176 GB in all) and the zip deleted. A rerun skips years already
extracted, so fetch resumes where it stopped.

build_zarr writes the store locally, from scratch on every run (~7 min a year).
publish uploads it to OUTPUT, replacing any published copy, and only runs when
named. It starts over if interrupted (46 GB, ~73k files). To finish an
interrupted upload, or to update only what changed after a rebuild, sync with
rclone instead (rclone sync <LOCAL> s3:<bucket/path of ZARR> --checksum).
"""

import http.client
import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  registers .rio
import xarray as xr
from zarr.codecs import ZstdCodec
from zarr.storage import ObjectStore

from cdh_data_pipeline import (
    download,
    log,
    open_raster,
    open_store,
    run,
    upload_dir,
    write_zarr,
)

SOURCE = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A"
ZIP = "{year}/0_Mosaicked/v1/1km_ua/global_agesex_structures_{year}_CN_1km_R2025A_UA_v1.zip"
TIF = "global_{sex}_{age:02d}_{year}_CN_1km_R2025A_UA_v1.tif"
NAME = "worldpop-age-sex-1km-ua-r2025a-v1"
# INPUT holds the zips in flight and one folder of GeoTIFFs per year.
# Gitignored under input/.
INPUT = "input/worldpop-age-sex"
# The store is built here, then uploaded by publish. Gitignored under output/.
LOCAL = f"output/{NAME}/{NAME}.zarr"
OUTPUT = f"s3://digital-atlas/cdh/data/{NAME}"
ZARR = f"{OUTPUT}/{NAME}.zarr"
# Zips downloading at once. The server limits per connection, not in total.
PREFETCH = 4
# Attempts per zip before fetch gives up on it.
RETRIES = 3

YEARS = list(range(2015, 2031))
# Lower bound of each age group in years: 0 (under 1), 1-4, 5-9, ..., 90+.
AGES = [0, 1, *range(5, 91, 5)]
AGE_LABELS = ["0", *(f"{a}-{b - 1}" for a, b in pairwise(AGES[1:])), "90+"]
# The zip's "t" files are f + m per age group, so they are left out.
SEXES = {"f": "female", "m": "male"}

# 30 arc-second grid, 84N to 60S. The source transform rounds the cell size to
# 0.0083333333, which drifts 1.4e-6 degrees by the east edge; coords are snapped
# to exact 1/120 so they align with other 30 arc-second datasets.
RES = 1 / 120
WIDTH, HEIGHT = 43200, 17280

# Expected reads aggregate many age/sex layers over an area: "total population
# of Kenya" (all 40 layers) or "working-age women in Kenya" (10 layers). So a
# chunk holds every age and sex for one year, and an area read costs the same
# few requests whatever layers it needs. 240 cells is 2 degrees (~220 km);
# a chunk is 9 MB uncompressed, 0.2-2 MB compressed.
# Not sharded: GDAL reads sharded zarr only from 3.13, and CRAN's R binaries ship
# 3.12 (Windows) and 3.8 (macOS). GDAL 3.12 reads this store. Measured on 2020
# over Kenya, a total took 30 requests (46 MB); with 9x9-chunk shards it took 13,
# and one layer per chunk took 160 (94 MB). Unsharded is ~4,600 objects a year
# instead of ~124. Adding shards back is a rebuild or a repack with no change to
# the chunks.
CHUNKS = (1, len(AGES), len(SEXES), 240, 240)
# Dask block: 9x9 chunks (18 degrees), so the grid splits into exactly 8x20
# blocks and no two parallel writes touch the same chunk.
BLOCK = 2160

ENCODING = {
    "population": {
        "chunks": CHUNKS,
        # Stored as int32 hundredths of a person. Measured on one layer: float32
        # is 2.7x larger, and float32 bit-rounded to 7 bits (0.4% error) 1.5x.
        # The global sum shifts by 1e-5, a cell by at most 0.005.
        "dtype": "int32",
        "scale_factor": 0.01,
        "_FillValue": -1,
        # zarr's own fill: unwritten and all-nodata chunks then read as nodata.
        "fill_value": -1,
        # zstd 9 matched blosc-zstd with bitshuffle and beat byte shuffle by 20%.
        # Level 15 is 7% smaller but 3x slower to write.
        "compressors": (ZstdCodec(level=9),),
    }
}


def tif_path(year, age, sex):
    return Path(INPUT) / str(year) / TIF.format(year=year, age=age, sex=sex)


def extract(zip_path, year):
    """Extract the year's female and male GeoTIFFs, skipping any already there.

    GDAL can read inside the zip, but each TIF is deflated, so every windowed
    read re-inflates from the start of the file.
    """
    (Path(INPUT) / str(year)).mkdir(parents=True, exist_ok=True)
    todo = [p for a in AGES for s in SEXES if not (p := tif_path(year, a, s)).exists()]

    def one(path):
        # Written to .part first, so a cut-off extract never looks finished.
        part = path.with_name(path.name + ".part")
        with (
            zipfile.ZipFile(zip_path) as z,
            z.open(path.name) as src,
            open(part, "wb") as f,
        ):
            while buf := src.read(1 << 24):
                f.write(buf)
        part.rename(path)

    # zlib releases the GIL, so threads inflate in parallel.
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, todo))


def fetch_year(year):
    """Download and extract one year, unless its GeoTIFFs are all present."""
    if all(tif_path(year, a, s).exists() for a in AGES for s in SEXES):
        return
    url = f"{SOURCE}/{ZIP.format(year=year)}"
    # The server sometimes drops a connection mid-download (seen on 4 of 15
    # zips, some after 40 MB); download() then raises, so retry from scratch.
    for attempt in range(1, RETRIES + 1):
        try:
            zip_path = download(url, f"{INPUT}/{year}.zip")
            break
        except (OSError, http.client.HTTPException) as e:
            log.warning("%d: attempt %d/%d failed: %s", year, attempt, RETRIES, e)
            if attempt == RETRIES:
                raise
    try:
        extract(zip_path, year)
    except zipfile.BadZipFile:
        # Truncated or corrupt (a CRC mismatch raises this too): drop it so a
        # rerun downloads it again instead of failing on the same file.
        zip_path.unlink()
        raise
    zip_path.unlink()
    log.info("fetched %d", year)


def fetch():
    """Download and extract every year, PREFETCH at a time."""
    with ThreadPoolExecutor(PREFETCH) as pool:
        list(pool.map(fetch_year, YEARS))


def open_year(year):
    """One year as a lazy population(year, age_group, sex, y, x) dataset."""

    def layer(age, sex):
        # lock=False: decoding the 40 LZW TIFs was the bottleneck under the
        # default lock (3.8x slower reads, ~2.4 of 16 cores busy).
        chunks = {"y": BLOCK, "x": BLOCK}
        return open_raster(str(tif_path(year, age, sex)), chunks=chunks, lock=False)

    # join="override": take the first raster's grid, skip coordinate alignment.
    da = xr.concat(
        [
            xr.concat([layer(age, sex) for sex in SEXES], "sex", join="override")
            for age in AGES
        ],
        "age_group",
        join="override",
    )
    assert da.sizes["x"] == WIDTH and da.sizes["y"] == HEIGHT, da.sizes
    # Every age and sex in each dask block, matching the zarr chunks.
    da = da.chunk({"age_group": -1, "sex": -1})
    da = da.expand_dims(year=[year]).assign_coords(
        x=-180 + (np.arange(WIDTH) + 0.5) * RES,
        y=84 - (np.arange(HEIGHT) + 0.5) * RES,
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
    """Write every year into the local store, from scratch."""
    for i, year in enumerate(YEARS):
        ds = open_year(year)
        if i == 0:
            # Empty store covering every year. Each year then fills its own slot.
            write_zarr(ds.reindex(year=YEARS), LOCAL, ENCODING, compute=False)
        # A region write only accepts variables on the region's dimension.
        ds = ds.drop_vars([c for c in ds.coords if c != "year"])
        ds.to_zarr(ObjectStore(open_store(LOCAL)), region="auto")
        log.info("wrote %d (%d/%d)", year, i + 1, len(YEARS))


def publish():
    """Upload the local store to OUTPUT, replacing any published copy."""
    upload_dir(LOCAL, ZARR)


if __name__ == "__main__":
    run(fetch, build_zarr, optional=(publish,))
