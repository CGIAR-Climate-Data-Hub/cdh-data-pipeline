"""CHIRPS v3.0 daily precipitation, RNL flavour (0.05 deg, 1981-present) -> Zarr cube.

Run from the repo root: uv run --env-file .env recipes/chirps_era5.py
The same command does the first build and every update. To rebuild from scratch,
delete the store and run it again. Individual steps:

    uv run recipes/chirps_era5.py update          # skip the archive build
    uv run recipes/chirps_era5.py verify_archive  # check the source rasters first
    uv run recipes/chirps_era5.py detect          # which flavour does the cube hold?

Not the same dataset as ``recipes/chirps.py``
---------------------------------------------
That recipe builds ``chirps-v3-rnl`` from the yearly NetCDFs on CHC's FTP mirror,
sharded (182, 256, 256) with Blosc, for dask/xarray region reads. This one builds
``chirps-v3-daily-rnl`` from the **daily GeoTIFF** archive, unsharded
(128, 300, 300) with plain Zstandard, so GDAL can read it -- QGIS, R's
terra/stars, and anything else outside the Python stack. Same upstream product,
two renditions with different read patterns; they live under different dataset
ids and neither is a replacement for the other.

Two sources, deliberately
-------------------------
* **The initial build reads a local GeoTIFF archive** (``CHIRPS_ARCHIVE``), laid
  out as ``<root>/<year>/chirps-v3.0.<yyyy>.<mm>.<dd>.tif``. 1981-present is
  ~16,600 days at ~2 MB/day stored; pulling that from CHC over HTTPS at their
  politeness limit would take weeks.
* **Every update reads CHC directly**, one directory listing per year, and
  downloads only the days whose upstream signature changed.

RNL and SAT are different products
----------------------------------
CHIRPS is fundamentally a *pentad* product; the daily values are a
disaggregation of pentad totals, and CHC publishes two of them:

=========  =====================  ==============
flavour    disaggregated with     record starts
=========  =====================  ==============
``sat``    NASA IMERG Late V07    1998
``rnl``    ECMWF ERA5             1981
=========  =====================  ==============

They are not interchangeable -- on a 2001-2002 sample, ``sat`` matches to 0.05 mm
where ``rnl`` differs by 178.8 mm. Appending one onto the other would blend two
records that look alike and are not, so the flavour is **verified against real
rasters** before any state is recorded, never assumed. See :func:`detect_flavour`.

Unlike CHIRTS, this gates on complete months: CHC publishes CHIRPS a whole month
at a time, 11-20 days after the month ends, and the lag is not dependable
(March 2026 arrived 57 days late). Ingesting a half-published month would record
it as done.

Readers should only trust a store whose root attr ``build_complete`` is true.
"""

import os
from pathlib import Path

import numpy as np

from cdh_data_pipeline import (
    PRECIP,
    GridSpec,
    RasterCubeBuilder,
    cube_store,
    log,
    run,
    write_json,
)
from cdh_data_pipeline.updater import ChirpsArchive, CubeUpdater
from cdh_data_pipeline.zarr import EPOCH

# Local GeoTIFF archive for the initial build. Gitignored under input/.
ARCHIVE = os.environ.get("CHIRPS_ARCHIVE", "input/chirps-v3-daily")
# Local while testing; publish to s3://digital-atlas/cdh/data/chirps-v3-daily-rnl.
OUTPUT = os.environ.get("CHIRPS_DAILY_OUTPUT", "output/chirps-v3-daily-rnl")
STORE_NAME = "chirps-v3-daily-rnl.zarr"
URL = f"{OUTPUT}/{STORE_NAME}"

#: The production archive. ``sat`` is a different product, not newer coverage.
FLAVOUR = "rnl"

#: First year of the TIME AXIS, not merely a filter: a Zarr time axis only grows
#: forward, so days before this cannot be backfilled without a full rebuild.
INIT_YEAR = 1981

# Threads for decoding local rasters. Nothing to do with the server.
BUILD_WORKERS = 6
# Concurrent downloads from CHC. Keep at 1: CHC bans clients that pull hard, and
# the shared rate limiter caps the request rate regardless of this number.
FETCH_WORKERS = 1
# Minimum seconds between requests to CHC.
MIN_INTERVAL = 1.0

VARIABLES = [PRECIP]

#: 0.05 deg, 60 N to 60 S. Narrower than CHIRTS, which reaches 70 N -- the two
#: grids are not interchangeable despite sharing a resolution.
CHIRPS_GRID = GridSpec(height=2400, width=7200, res=0.05, north=60.0, nodata=-9999.0)

ATTRS = {
    "title": "CHIRPS v3.0 global daily precipitation (RNL)",
    "institution": "Climate Hazards Center, University of California, Santa Barbara",
    "source": "CHIRPS v3.0 daily final RNL (ERA5-disaggregated pentads), 0.05 deg",
    "references": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/readme.txt",
}


def _builder() -> RasterCubeBuilder:
    return RasterCubeBuilder(
        ARCHIVE,
        grid=CHIRPS_GRID,
        variables=VARIABLES,
        init_year=INIT_YEAR,
        workers=BUILD_WORKERS,
        global_attrs=ATTRS,
    )


def _updater(verify=None) -> CubeUpdater:
    return CubeUpdater(
        URL,
        remote=ChirpsArchive(flavour=FLAVOUR, min_interval=MIN_INTERVAL),
        variables=VARIABLES,
        grid=CHIRPS_GRID,
        # CHIRPS drops a whole month at a time; see the module docstring.
        require_complete_months=True,
        workers=FETCH_WORKERS,
        verify=verify,
        extra_state={
            "mission": "chirps-v3-daily",
            "flavour": FLAVOUR,
            "archive": ChirpsArchive.BASE,
        },
    )


def _built() -> bool:
    """True if a finished store is already there.

    An existence probe, so any failure to open means "not built" -- a half-written
    store left by an interrupted run reads as absent and gets rebuilt, which is
    the safe direction.
    """
    import zarr

    try:
        root = zarr.open_group(cube_store(URL), mode="r", use_consolidated=False)
    except Exception:  # noqa: BLE001  see docstring
        return False
    return bool(root.attrs.get("build_complete"))


def detect_flavour(candidates=("sat", "rnl"), n_days=3, cache="input/_flavour") -> dict:
    """Which CHIRPS flavour the cube's values actually came from.

    The flavours are the same pentads disaggregated differently -- IMERG vs ERA5 --
    so the one that matches to within the quantisation bound is the one the cube
    was built from, and the others disagree by millimetres.

    This exists because the signature comparison cannot catch a *wrongly recorded*
    state: recording writes down whatever flavour it is given, so an unverified
    guess becomes a wrong label that later updates act on.
    """
    import rasterio
    import zarr

    root = zarr.open_group(cube_store(URL), mode="r", use_consolidated=False)
    days = (
        (EPOCH + np.asarray(root["time"][:]).astype("timedelta64[D]"))
        .astype(object)
        .tolist()
    )
    sample = np.asarray(root[PRECIP.nc_name][:, ::101, ::401])
    present = [
        d for d, ok in zip(days, (sample != PRECIP.fill).any(axis=(1, 2))) if ok
    ]
    if not present:
        raise RuntimeError("cube holds no data to compare against")
    # Sample from the late record: every flavour covers it, unlike the 1980s.
    picks = [present[int(len(present) * f)] for f in (0.6, 0.75, 0.9)][:n_days]
    index = {d: i for i, d in enumerate(days)}

    results = {}
    for flavour in candidates:
        remote = ChirpsArchive(flavour=flavour, min_interval=MIN_INTERVAL)
        errs, checked = [], 0
        for day in picks:
            listing = {r.day: r for r in remote.list_year("precip", day.year)}
            if day not in listing:
                continue
            path = remote.download(listing[day], Path(cache))
            with rasterio.open(path) as src:
                ref = src.read(1)
            got = PRECIP.decode(np.asarray(root[PRECIP.nc_name][index[day]]))
            both = (ref != CHIRPS_GRID.nodata) & ~np.isnan(got)
            if both.any():
                errs.append(float(np.abs(got[both] - ref[both]).max()))
                checked += 1
        results[flavour] = {
            "days_compared": checked,
            "max_abs_error_mm": max(errs) if errs else None,
            # Within half a quantum is a match; anything else is a different
            # product, not rounding.
            "matches": bool(errs) and max(errs) <= PRECIP.scale / 2 + 1e-3,
        }
        verdict = (
            "MATCH"
            if results[flavour]["matches"]
            else "no data for these days"
            if not checked
            else "differs"
        )
        err = results[flavour]["max_abs_error_mm"]
        log.info(
            "  %-4s %-24s max error %s",
            flavour, verdict, "n/a" if err is None else f"{err:.4f} mm",
        )
    matched = [f for f, r in results.items() if r["matches"]]
    return {
        "candidates": results,
        "detected": matched[0] if len(matched) == 1 else None,
    }


def _verify_flavour():
    """Prove the cube holds FLAVOUR before its days are recorded as ingested."""
    log.info("verifying the cube really holds CHIRPS %r data ...", FLAVOUR)
    found = detect_flavour()["detected"]
    if found is None:
        raise RuntimeError(
            "the cube matches no CHIRPS flavour tested; it may be a different "
            "product or version"
        )
    if found != FLAVOUR:
        raise ValueError(
            f"this cube holds CHIRPS {found!r} data, not {FLAVOUR!r}. Recording it "
            f"would label it wrongly and later updates would blend two different "
            f"products."
        )


def detect():
    """Report which flavour the cube actually holds, and stop."""
    log.info("%s", detect_flavour())


def verify_archive():
    """Check every source raster opens, decodes and has the expected grid.

    Not run by default -- it decodes the whole archive. Worth it before a first
    build: a TIFF whose directory is intact while its strips are damaged opens
    perfectly and fails only at read(), hours into a build.
    """
    report = _builder().verify_archive()
    if report["unreadable"]:
        log.warning(
            "%d unreadable file(s); those days would land as nodata and the "
            "update step would refetch them from CHC",
            len(report["unreadable"]),
        )


def build_zarr():
    """Build the cube from the local archive, then record what it holds.

    Skipped when a finished store already exists -- the update step carries it
    forward from there. Recording is not optional: without it every day looks
    un-ingested and the first update would try to refetch the entire record from
    CHC. It runs the flavour check first.
    """
    if _built():
        log.info("%s already built; skipping to update", URL)
        return
    result = _builder().run(URL)
    log.info(
        "built %s: %d day(s), %.2f GB", URL, result["days"], result["bytes"] / 1e9
    )
    if result["unreadable"]:
        log.warning(
            "%d day(s) left nodata by unreadable source rasters; the update step "
            "refetches them from CHC",
            len(result["unreadable"]),
        )
    log.info("recording upstream signatures for the days just built")
    _updater(verify=_verify_flavour).record_existing()


def update():
    """Fetch and write the days CHC has that the store does not."""
    result = _updater().update()
    log.info("update: %s", result.get("plan", ""))
    if result.get("written"):
        log.info(
            "wrote %d day(s); coverage now %s",
            result["written"], " -> ".join(result["coverage"]),
        )


def write_sources():
    """Publish the upstream manifest beside the store.

    Per-day signatures live in ``update-state.json``; this is the dataset-level
    summary CONVENTIONS.md asks every dataset to carry.
    """
    upd = _updater()
    state = upd.load_state()
    write_json(
        f"{OUTPUT}/sources.json",
        {
            "dataset": "chirps-v3-daily-rnl",
            "archive": ChirpsArchive.BASE,
            "flavour": FLAVOUR,
            "retrieved": state.get("updated") or state.get("recorded"),
            "day_state": "update-state.json",
            "days_ingested": {k: len(v) for k, v in state.get("days", {}).items()},
            "coverage": [str(d) for d in upd.cube_span()],
        },
    )


if __name__ == "__main__":
    run(build_zarr, update, write_sources)
