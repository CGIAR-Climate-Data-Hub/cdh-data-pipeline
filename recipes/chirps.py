"""CHIRPS v3.0 daily precipitation (0.05 deg, 1981-present) -> sharded Zarr cube.

Run from the repo root: uv run --env-file .env recipes/chirps.py
The same command does the first build and every update. To rebuild from scratch,
delete the store and run it again.

CHC asks scripted downloads to use FTP, not https://data.chc.ucsb.edu (see the
notice on that page), so this recipe pulls from their FTP mirror. Anonymous login,
passive mode, ~46 x 4 GB files. One stream runs at roughly 2 MB/s (a day for
everything), so WORKERS files download at once. Resumable at file granularity.

Updates are incremental. Both steps compare against the store's ``sources`` attr
(authoritative; sources.json is a copy for people), so any machine can update the
store from an empty cache, downloading only the years it lacks. Readers should only
trust a store whose root attr ``build_complete`` is true.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask
import numpy as np
import rioxarray  # noqa: F401  registers .rio
import xarray as xr
import zarr
from zarr.storage import ObjectStore

from cdh_data_pipeline import (
    blosc_zstd,
    check_packable,
    download,
    ftp_files,
    log,
    open_store,
    read_manifest,
    run,
    write_json,
    write_zarr,
)

# INPUT is the local NetCDF cache. Gitignored under input/.
INPUT = "input/chirps_v3"
# Local while testing; publish to s3://digital-atlas/cdh/data/chirps-v3-rnl.
OUTPUT = "output/chirps-v3-rnl"
STORE_NAME = "chirps-v3-rnl.zarr"
# Concurrent FTP connections. 8 tested fine (~20 MB/s aggregate, no per-IP cap hit).
# Lower it if the server answers "421 too many connections".
WORKERS = 8
# Dask processes for writes, not threads: h5py serializes HDF5 decompression
# within a process. Each holds a few 182-day x 256-row stripes (~1.3 GB).
BUILD_WORKERS = 8

FTP = "ftp://ftp.chc.ucsb.edu/pub/org/chc/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byYear"

# Expected reads are point or small-region series of ~200 days to a year, with
# points clustered by region. (182, 32, 32) chunks answer those in 2-6 requests;
# (182, 256, 256) shards (12.8 deg) cap the store at ~27k objects. int16 tenths of
# a mm: the 1981-2026 cache peaks at ~1470 mm/day against a 3276.7 limit.
ENCODING = {
    "dtype": "int16",
    "scale_factor": 0.1,
    "_FillValue": -32768,
    "chunks": (182, 32, 32),
    "shards": (182, 256, 256),
    "compressors": (blosc_zstd(2, 3, shuffle="bitshuffle"),),
}
TIME_ENCODING = {
    "units": "days since 1980-01-01",
    "calendar": "standard",
    "dtype": "int32",
}
ENCODINGS = {"precip": ENCODING, "time": TIME_ENCODING}

ATTRS = {
    "title": "CHIRPS v3.0 daily precipitation (RNL)",
    "institution": "Climate Hazards Center, University of California, Santa Barbara",
    "source": "CHIRPS v3.0 daily final RNL (ERA5-disaggregated pentads), 0.05 deg",
    "references": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/readme.txt",
}
PRECIP_ATTRS = {"long_name": "Daily precipitation", "units": "mm/day"}


def open_group(url, mode="r+"):
    """Open the store's root group, bypassing possibly stale consolidated metadata."""
    return zarr.open_group(
        ObjectStore(open_store(url)), mode=mode, use_consolidated=False
    )


def built_sources(url):
    """The ``sources`` attr of the last completed build; {} if there is none."""
    try:
        group = open_group(url, "r")
    except FileNotFoundError:
        return {}
    if group["precip"].shards != ENCODING["shards"]:
        raise ValueError(f"{url} was written with other shards; delete it to rebuild")
    return group.attrs.get("sources", {})


def fetch():
    """Download the upstream file versions the store lacks (all, if no store yet).

    Not a full mirror: years already in the store are skipped, and download() skips
    files already cached at their current version. Needs access to OUTPUT.
    """
    remote = ftp_files(FTP, ".nc")
    built = built_sources(f"{OUTPUT}/{STORE_NAME}")
    todo = [n for n, v in remote.items() if built.get(n, {}).get("version") != v]
    log.info("store lacks %d of %d files", len(todo), len(remote))

    def fetch_one(name):
        download(f"{FTP}/{name}", Path(INPUT, name), version=remote[name])

    # Each urllib FTP request opens its own connection, so threads don't share state.
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(fetch_one, todo))  # list() re-raises a failed download


def write_file(url, name, new):
    """Write one cached yearly file into the store at its dates.

    ``new`` creates the store from this file. Otherwise days the store already has
    are overwritten in place and later days are appended. Dask chunks are the
    store's shards, so each task writes whole shards (Zarr merges a partial one).
    """
    with xr.open_dataset(Path(INPUT, name), engine="h5netcdf") as src:
        # Sources are south-up; rounding drops float32 noise (59.874992).
        ds = src.rename(latitude="y", longitude="x").isel(y=slice(None, None, -1))
        ds = ds.assign_coords(
            y=ds.y.astype("float64").round(3), x=ds.x.astype("float64").round(3)
        )
        ds.precip.attrs, ds.precip.encoding = PRECIP_ATTRS, {}
        ds = ds.assign_attrs(ATTRS, build_complete=False).rio.write_crs("EPSG:4326")
        step, rows, _ = ENCODING["shards"]
        if new:
            write_zarr(ds.chunk(time=step, y=rows, x=-1), url, ENCODINGS)
            return
        target = ObjectStore(open_store(url))
        store = xr.open_zarr(target, consolidated=False)
        xr.align(store, ds, join="exact", exclude="time")  # raises if the grid moved
        times = store.get_index("time")
        start = times.searchsorted(ds.time.values[0])
        if start == len(times) and ds.time[0] != times[-1] + np.timedelta64(1, "D"):
            raise ValueError(f"{name} does not continue the days in {url}")
        # Dask chunks on the store's time shard boundaries: each task writes whole
        # shards, and only the first and last can be partial (Zarr merges those).
        stop = start + ds.sizes["time"]
        edges = [start, *range((start // step + 1) * step, stop, step), stop]
        ds = ds.chunk(time=tuple(np.diff(edges)), y=rows, x=-1)
        ds = check_packable(ds, ENCODINGS).drop_vars(["y", "x", "spatial_ref"])
        split = int(ds.time.isin(times).sum())  # leading days the store already has
        kwargs = {"zarr_format": 3, "consolidated": False, "align_chunks": True}
        if split:
            ds.isel(time=slice(None, split)).to_zarr(target, region="auto", **kwargs)
        if split < ds.sizes["time"]:
            ds.isel(time=slice(split, None)).to_zarr(
                target, append_dim="time", **kwargs
            )


def build_zarr():
    """Write cached files whose content the store lacks, then publish sources.json."""
    url = f"{OUTPUT}/{STORE_NAME}"
    cached = {n: e for n, e in read_manifest(INPUT).items() if Path(INPUT, n).exists()}
    built = built_sources(url)
    todo = sorted(
        n for n, e in cached.items() if built.get(n, {}).get("sha256") != e["sha256"]
    )
    if not todo and not built:
        raise ValueError(f"no CHIRPS files in {INPUT}; run fetch first")
    if todo:
        log.info(
            "%s %s with %d files", "updating" if built else "building", url, len(todo)
        )
        if built:
            open_group(url).attrs["build_complete"] = False
        with dask.config.set(scheduler="processes", num_workers=BUILD_WORKERS):
            for i, name in enumerate(todo):
                log.info("writing %s", name)
                write_file(url, name, new=not built and i == 0)
        sources = {**built, **{n: cached[n] for n in todo}}
        open_group(url).attrs.update(sources=sources, build_complete=True)
        zarr.consolidate_metadata(ObjectStore(open_store(url)))
        write_json(f"{OUTPUT}/sources.json", sources)
    else:
        log.info("store is up to date with %s", INPUT)


if __name__ == "__main__":
    run(fetch, build_zarr)
