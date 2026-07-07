"""Recipe entrypoint helpers."""

import os
import sys


def run(*builders):
    """Run build steps, flush output, then hard-exit.

    zarr v3 and obstore can leave noisy async teardown at interpreter shutdown.
    """
    for build in builders:
        build()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
