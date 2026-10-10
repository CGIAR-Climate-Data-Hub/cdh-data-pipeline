"""CHIRTS-ERA5 daily Tmax/Tmin (0.05 deg, 1981-present) -> Zarr cube.

Run from the repo root: uv run --env-file .env recipes/chirts_era5.py
The same command does the first build and every update. To rebuild from scratch,
delete the store and run it again. Individual steps:

    uv run recipes/chirts_era5.py update          # skip the archive build
    uv run recipes/chirts_era5.py verify_archive  # check the source rasters first

Two sources, deliberately:

* **The initial build reads a local GeoTIFF archive** (``CHIRTS_ARCHIVE``, a
  mounted share). 1981-present is ~16,600 days x 2 variables at ~26 MB/day/var;
  pulling that from CHC over HTTPS at their politeness limit would take weeks.
* **Every update reads CHC directly**, one directory listing per variable per
  year, and downloads only the days whose upstream signature changed.

Grid facts, verified against a 2026 raster from the server: 7200 x 2600 at
0.05 deg, -180..180 lon, **-60..70 lat** -- CHIRTS reaches 70 N where CHIRPS
stops at 60 N, so the two grids are *not* interchangeable despite sharing a
resolution. float32 degrees Celsius (not Kelvin, unlike AgERA5), nodata -9999
undeclared in the GeoTIFF profile, ~69% nodata, LZW striped one row per block.

Why days rather than whole months: CHIRTS-ERA5 trickles individual days out at a
~5-day lag, so gating on complete months would hold back up to a month of usable
data. CHIRPS, which drops a month at a time, does gate.

Readers should only trust a store whose root attr ``build_complete`` is true.
"""

import os

from cdh_data_pipeline import (
    TMAX,
    TMIN,
    GridSpec,
    RasterCubeBuilder,
    cube_store,
    log,
    run,
    write_json,
)
from cdh_data_pipeline.updater import ChirtsArchive, CubeUpdater

# Local GeoTIFF archive for the initial build. Gitignored under input/.
ARCHIVE = os.environ.get("CHIRTS_ARCHIVE", "input/chirts-era5")
# Local while testing; publish to s3://digital-atlas/cdh/data/chirts-era5-daily.
OUTPUT = os.environ.get("CHIRTS_OUTPUT", "output/chirts-era5-daily")
STORE_NAME = "chirts-era5-daily.zarr"
URL = f"{OUTPUT}/{STORE_NAME}"

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

VARIABLES = [TMAX, TMIN]

CHIRTS_GRID = GridSpec(
    height=2600, width=7200, res=0.05, north=70.0, nodata=-9999.0
)

ATTRS = {
    "title": "CHIRTS-ERA5 daily maximum and minimum temperature",
    "institution": "Climate Hazards Center, University of California, Santa Barbara",
    "source": (
        "CHIRTS-ERA5 daily GeoTIFFs "
        "(https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/)"
    ),
    "references": "https://doi.org/10.1038/s41597-020-00643-7",
}


def _builder() -> RasterCubeBuilder:
    return RasterCubeBuilder(
        ARCHIVE,
        grid=CHIRTS_GRID,
        variables=VARIABLES,
        init_year=INIT_YEAR,
        workers=BUILD_WORKERS,
        global_attrs=ATTRS,
    )


def _updater() -> CubeUpdater:
    return CubeUpdater(
        URL,
        remote=ChirtsArchive(min_interval=MIN_INTERVAL),
        variables=VARIABLES,
        grid=CHIRTS_GRID,
        # CHIRTS publishes day by day; see the module docstring.
        require_complete_months=False,
        workers=FETCH_WORKERS,
        extra_state={"mission": "chirts-era5", "archive": ChirtsArchive.BASE},
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
    CHC.
    """
    if _built():
        log.info("%s already built; skipping to update", URL)
        return
    builder = _builder()
    result = builder.run(URL)
    log.info(
        "built %s: %d day(s), %.2f GB",
        URL, result["days"], result["bytes"] / 1e9,
    )
    if result["unreadable"]:
        log.warning(
            "%d day(s) left nodata by unreadable source rasters; the update step "
            "refetches them from CHC",
            len(result["unreadable"]),
        )
    log.info("recording upstream signatures for the days just built")
    _updater().record_existing()


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
    days = state.get("days", {})
    write_json(
        f"{OUTPUT}/sources.json",
        {
            "dataset": "chirts-era5-daily",
            "archive": ChirtsArchive.BASE,
            "retrieved": state.get("updated") or state.get("recorded"),
            "day_state": "update-state.json",
            "days_ingested": {k: len(v) for k, v in days.items()},
            "coverage": [str(d) for d in upd.cube_span()],
        },
    )


if __name__ == "__main__":
    run(build_zarr, update, write_sources)
