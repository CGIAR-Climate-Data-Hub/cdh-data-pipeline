"""Recipe entrypoint: run a recipe's build steps, then clean up."""

import os
import sys


def run(*builders):
    """Run each build step in order, then hard-exit.

    Call from a recipe's __main__: ``run(build_zarr, build_cogs)`` -- pass as many
    builders as the recipe has (e.g. two zarr stores with different chunking). Once
    they return the work is flushed; zarr v3 + obstore leave noisy async threads at
    shutdown, so we os._exit rather than sit through the messy teardown.
    """
    for build in builders:
        build()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
