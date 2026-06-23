"""COG conversion: re-encode a source raster as a self-describing Cloud-Optimized GeoTIFF."""

from concurrent.futures import ThreadPoolExecutor

import rasterio
from rasterio.io import MemoryFile

COG_OPTS = dict(
    driver="COG",
    compress="ZSTD",
    level=22,
    predictor="YES",
    blocksize=512,
    overview_resampling="average",
)


def make_cog(src_url, desc, units):
    """Re-encode one source raster as a COG in memory; returns the file bytes."""
    with rasterio.open(src_url) as s:
        profile = {**s.profile, **COG_OPTS}
        for k in ("blockxsize", "blockysize", "tiled", "interleave"):
            profile.pop(k, None)
        data = s.read()
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data)
            dst.descriptions = (desc,)
            dst.units = (units,)
        return mem.read()


def write_cogs(store, jobs, *, workers=1, log_every=0):
    """Build + store a batch of COGs.

    jobs: iterable of (out_name, src_url, desc, units). workers>1 builds them in a
    thread pool; log_every>0 prints 'i/total' progress. Returns the out_names written.
    """
    jobs = list(jobs)

    def one(job):
        out, src_url, desc, units = job
        store.put(out, make_cog(src_url, desc, units))
        return out

    done = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(one, jobs), 1):
            done.append(out)
            if log_every and (i % log_every == 0 or i == len(jobs)):
                print(f"  cog {i}/{len(jobs)}")
    return done
