"""Recipe entrypoint helpers and the shared logger."""

import logging
import os
import sys
import time

log = logging.getLogger("cdh")


def _timed(name, step):
    """Run one step and log how long it took."""
    log.info("== %s", name)
    t0 = time.monotonic()
    try:
        step()
    except Exception:
        log.exception("== %s failed after %.0fs", name, time.monotonic() - t0)
        raise
    log.info("== %s done in %.0fs", name, time.monotonic() - t0)


def _exit(code):
    """Hard-exit to skip zarr/obstore's noisy async teardown."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run(*builders, optional=()):
    """Run build steps in order; exit 1 on the first failure.

    Pass step names to run a subset: ``uv run recipes/mapspam.py build_cogs``.
    Steps in ``optional``, such as a ``publish`` upload, run only when named.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    steps = {b.__name__: b for b in (*builders, *optional)}
    names = sys.argv[1:] or [b.__name__ for b in builders]
    if unknown := set(names) - steps.keys():
        sys.exit(f"unknown step(s) {sorted(unknown)}; choose from {list(steps)}")
    try:
        for name in names:
            _timed(name, steps[name])
    except Exception:  # noqa: BLE001  already logged
        _exit(1)
    _exit(0)
