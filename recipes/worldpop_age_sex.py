"""WorldPop age and sex structures (R2025A, 1 km, UN-adjusted) -> Zarr store.

Run from the repo root: uv run recipes/worldpop_age_sex.py

The source is one 15.6 GB zip per year (250 GB in all). The server ignores
range requests and caps each connection at ~5-9 MB/s, so PREFETCH zips download
in parallel while the current year is written. Each zip's 40 female and male
GeoTIFFs are extracted (11 GB), written into the store as one year, then
deleted. Local disk use peaks near (PREFETCH + 1) x 16 GB + 11 GB.

A rerun skips the years listed in INPUT/done.txt. Delete that file to rebuild
from scratch.
"""

import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  registers .rio
import xarray as xr
from zarr.codecs import ZstdCodec
from zarr.storage import ObjectStore

from cdh_data_pipeline import download, log, open_raster, open_store, run, write_zarr

SOURCE = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A"
ZIP = "{year}/0_Mosaicked/v1/1km_ua/global_agesex_structures_{year}_CN_1km_R2025A_UA_v1.zip"
TIF = "global_{sex}_{age:02d}_{year}_CN_1km_R2025A_UA_v1.tif"
# INPUT holds the zips in flight, the extracted year and the resume log.
# Gitignored under input/.
INPUT = "input/worldpop-age-sex"
OUTPUT = "s3://digital-atlas/cdh/data/worldpop-age-sex-1km-ua-r2025a-v1"
ZARR = f"{OUTPUT}/worldpop-age-sex-1km-ua-r2025a-v1.zarr"
# Zips downloading at once. The server limits per connection, not in total.
PREFETCH = 4

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
# few requests whatever layers it needs. Measured on 2020 over Kenya, a total
# took 13 requests (46 MB) where one layer per chunk took 160 (94 MB).
# 240 cells is 2 degrees (~220 km); a shard of 9x9 chunks is 18 degrees and the
# grid splits into exactly 8x20 shards. A chunk is 9 MB uncompressed.
CHUNKS = (1, len(AGES), len(SEXES), 240, 240)
SHARDS = (1, len(AGES), len(SEXES), 2160, 2160)

ENCODING = {
    "population": {
        "chunks": CHUNKS,
        "shards": SHARDS,
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


def fetch(year):
    """Download one year's zip. Returns its local path."""
    return download(f"{SOURCE}/{ZIP.format(year=year)}", f"{INPUT}/{year}.zip")


def extract(zip_path, year):
    """Extract the year's female and male GeoTIFFs. Returns their directory.

    GDAL can read inside the zip, but each TIF is deflated, so every windowed
    read re-inflates from the start of the file.
    """
    dest = Path(INPUT) / str(year)
    dest.mkdir(exist_ok=True)
    names = [TIF.format(year=year, age=a, sex=s) for a in AGES for s in SEXES]
    todo = [n for n in names if not (dest / n).exists()]

    def one(name):
        part = dest / f"{name}.part"
        with zipfile.ZipFile(zip_path) as z, z.open(name) as src, open(part, "wb") as f:
            while buf := src.read(1 << 24):
                f.write(buf)
        part.rename(dest / name)

    # zlib releases the GIL, so threads inflate in parallel.
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, todo))
    return dest


def open_year(tif_dir, year):
    """One year as a lazy population(year, age_group, sex, y, x) dataset."""

    def layer(age, sex):
        path = tif_dir / TIF.format(year=year, age=age, sex=sex)
        # lock=False: decoding the 40 LZW TIFs was the bottleneck under the
        # default lock (3.8x slower reads, ~2.4 of 16 cores busy).
        chunks = {"y": SHARDS[-2], "x": SHARDS[-1]}
        return open_raster(str(path), chunks=chunks, lock=False)

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
    # One dask chunk per shard, so parallel writes never share a shard.
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
    done_log = Path(INPUT) / "done.txt"
    done = done_log.read_text().split() if done_log.exists() else []
    todo = [y for y in YEARS if str(y) not in done]
    pool = ThreadPoolExecutor(PREFETCH)
    # Submitted in order, so the pool downloads the next PREFETCH years.
    zips = {year: pool.submit(fetch, year) for year in todo}
    try:
        for year in todo:
            zip_path = zips.pop(year).result()
            tif_dir = extract(zip_path, year)
            ds = open_year(tif_dir, year)
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
            for tif in tif_dir.iterdir():
                tif.unlink()
            tif_dir.rmdir()
            zip_path.unlink()
            log.info("wrote %d (%d/%d)", year, len(done), len(YEARS))
    finally:
        # On failure, drop queued downloads. Ones in flight still finish before exit.
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    run(build_zarr)
