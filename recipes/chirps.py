"""CHIRPS v3.0 daily precipitation (0.05 deg, 1981-present) -> local NetCDF cache.

Run from the repo root: uv run recipes/chirps.py

Only the download step exists so far. Rechunk/recompress to Zarr comes next.

CHC asks scripted downloads to use FTP, not https://data.chc.ucsb.edu (see the
notice on that page), so this recipe pulls from their FTP mirror. Anonymous login,
passive mode, ~46 x 4 GB files. One stream runs at roughly 2 MB/s (a day for
everything), so WORKERS files download at once. Resumable at file granularity.
"""

import ftplib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cdh_data_pipeline import download, log, run

# INPUT is the local NetCDF cache. Gitignored under input/.
INPUT = "input/chirps_v3"
# Concurrent FTP connections. 8 tested fine (~20 MB/s aggregate, no per-IP cap hit).
# Lower it if the server answers "421 too many connections".
WORKERS = 8

FTP_HOST = "ftp.chc.ucsb.edu"
FTP_DIR = "/pub/org/chc/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byYear"


def remote_files():
    """Return {filename: size} for the .nc files in FTP_DIR."""
    with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
        ftp.login()
        ftp.voidcmd("TYPE I")  # vsftpd refuses SIZE in ASCII mode
        paths = [p for p in ftp.nlst(FTP_DIR) if p.endswith(".nc")]
        return {Path(p).name: ftp.size(p) for p in paths}


def fetch():
    """Download every yearly NetCDF into INPUT.

    Files already present are skipped unless their size differs from the server's.
    CHC rewrites recent years as late station data arrives, so a size mismatch means
    the upstream file changed and we re-download it.
    """
    files = remote_files()
    log.info("%d files on server, %.0f GB", len(files), sum(files.values()) / 1e9)

    def fetch_one(name, size):
        dest = Path(INPUT) / name
        if dest.exists() and dest.stat().st_size != size:
            log.info("%s changed upstream, re-downloading", name)
            dest.unlink()
        download(f"ftp://{FTP_HOST}{FTP_DIR}/{name}", dest)

    # Each urllib FTP request opens its own connection, so threads don't share state.
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda kv: fetch_one(*kv), sorted(files.items())))


if __name__ == "__main__":
    run(fetch)
