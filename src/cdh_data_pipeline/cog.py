"""Cloud-Optimized GeoTIFF writer."""

import rasterio
from rasterio.io import MemoryFile

from cdh_data_pipeline.storage import open_store

COG_OPTS = dict(
    driver="COG",
    compress="ZSTD",
    level=22,
    predictor="YES",
    blocksize=512,
    bigtiff="IF_SAFER",
)


def make_cog(
    srcs,
    descriptions,
    units,
    *,
    long_names=None,
    overview_resampling="average",
    interleave="PIXEL",
):
    """Build a COG in memory and return its bytes.

    ``srcs`` and ``descriptions`` are per-band lists. Each source band is read as a
    full array, and GDAL also assembles the COG plus overviews in memory.

    Descriptions are what GDAL/terra expose as band names, so keep them short and
    stable (e.g. crop codes); ``long_names`` adds a human-readable ``long_name``
    metadata tag per band.

    ``interleave="BAND"`` keeps single-band reads cheap for multi-band COGs.
    """
    with rasterio.open(srcs[0]) as s0:
        profile = {
            **s0.profile,
            **COG_OPTS,
            "count": len(srcs),
            "overview_resampling": overview_resampling,
            "interleave": interleave,
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
    """Build a COG with :func:`make_cog` and write it to ``url``."""
    prefix, _, name = url.rpartition("/")
    open_store(prefix).put(name, make_cog(srcs, descriptions, units, **kwargs))
    print(f"wrote {url} ({len(srcs)} bands)")
