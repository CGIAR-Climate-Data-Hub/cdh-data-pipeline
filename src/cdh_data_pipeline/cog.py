"""Cloud-Optimized GeoTIFF writer."""

import rasterio
from rasterio.io import MemoryFile

from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import open_store

_COG_OPTS = {
    "driver": "COG",
    "compress": "ZSTD",
    "level": 9,
    "predictor": "YES",
    "blocksize": 512,
    "num_threads": "ALL_CPUS",
    "bigtiff": "IF_SAFER",
    "overview_resampling": "average",
    "interleave": "PIXEL",
}


def make_cog(
    srcs,
    descriptions,
    units,
    *,
    long_names=None,
    cog_options=None,
):
    """Build a multi-band COG in memory and return its bytes.

    One band per source file. ``descriptions`` become band names, so keep them
    short; ``long_names`` adds a readable ``long_name`` tag per band.
    ``cog_options`` overrides GDAL creation options, e.g. ``{"interleave": "BAND"}``.
    """
    with rasterio.open(srcs[0]) as s0:
        profile = {
            **s0.profile,
            **_COG_OPTS,
            **(cog_options or {}),
            "count": len(srcs),
        }
        for k in ("blockxsize", "blockysize", "tiled"):
            profile.pop(k, None)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            for i, u in enumerate(srcs, 1):
                with rasterio.open(u) as s:
                    dst.write(s.read(1), i)
            dst.descriptions = tuple(descriptions)
            dst.units = tuple([units] * len(srcs))
            if long_names:
                for i, ln in enumerate(long_names, 1):
                    dst.update_tags(i, long_name=ln)
        return mem.read()


def write_cog(url, srcs, descriptions, units, **kwargs):
    """Build a COG with ``make_cog`` and write it to ``url``."""
    prefix, _, name = url.rpartition("/")
    log.info("writing %s (%d bands)", url, len(srcs))
    open_store(prefix).put(name, make_cog(srcs, descriptions, units, **kwargs))
    log.info("wrote %s", url)
