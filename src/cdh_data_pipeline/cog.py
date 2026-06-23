"""COG conversion: re-encode a source raster as a self-describing Cloud-Optimized GeoTIFF."""

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
