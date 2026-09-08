"""Recipe entrypoint helpers and the shared logger."""

import logging
import os
import sys
import time

# One logger for the library and recipes. Timestamps stand in for progress bars:
# recipes run unattended and the slow parts (dask, GDAL) cannot report percent.
log = logging.getLogger("cdh")


def run(*builders):
    """Run build steps with timing, flush output, then hard-exit.

    zarr v3 and obstore can leave noisy async teardown at interpreter shutdown.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    for build in builders:
        log.info("== %s", build.__name__)
        t0 = time.monotonic()
        build()
        log.info("== %s done in %.0fs", build.__name__, time.monotonic() - t0)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
