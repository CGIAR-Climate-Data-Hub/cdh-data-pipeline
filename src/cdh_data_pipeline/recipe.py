"""Recipe entrypoint helpers and the shared logger."""

import logging
import os
import sys
import time

# One logger for the library and recipes. Timestamps stand in for progress bars:
# recipes run unattended and the slow parts (dask, GDAL) cannot report percent.
log = logging.getLogger("cdh")


def _timed(name, step):
    """Run one step, logging its name and elapsed time, also on failure."""
    log.info("== %s", name)
    t0 = time.monotonic()
    try:
        step()
    except Exception:
        log.exception("== %s failed after %.0fs", name, time.monotonic() - t0)
        raise
    log.info("== %s done in %.0fs", name, time.monotonic() - t0)


def _exit(code):
    """Flush and hard-exit; skips zarr v3 / obstore's noisy async teardown."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run(*builders):
    """Run build steps in order; exit 1 on the first failure.

    Step names on the command line select a subset, e.g.
    ``uv run recipes/mapspam.py build_cogs`` re-runs only that step.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    steps = {b.__name__: b for b in builders}
    names = sys.argv[1:] or list(steps)
    if unknown := set(names) - steps.keys():
        sys.exit(f"unknown step(s) {sorted(unknown)}; choose from {list(steps)}")
    try:
        for name in names:
            _timed(name, steps[name])
    except Exception:
        _exit(1)
    _exit(0)
