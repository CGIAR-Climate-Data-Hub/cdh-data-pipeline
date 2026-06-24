"""COG conversion: re-encode source raster(s) as a self-describing Cloud-Optimized GeoTIFF."""

import rasterio
from rasterio.io import MemoryFile

COG_OPTS = dict(
    driver="COG",
    compress="ZSTD",
    level=22,
    predictor="YES",
    blocksize=512,
)


def make_cog(
    srcs, descriptions, units, *, overview_resampling="average", interleave="PIXEL"
):
    """Re-encode single-band source raster(s) as one COG in memory; returns the bytes.

    srcs and descriptions are per-band lists (one band per source, in order) - pass a
    single source for a 1-band COG. Bands are read + written one at a time (no full array
    held), but GDAL assembles the whole COG + overviews in memory. In testing a 46-band
    global COG peaked ~2.5 GB, so call it per file and don't run these concurrently.

    overview_resampling: internal-overview method from GDAL: "average", "mode", "rms",
    "nearest", "bilinear", "cubic"
    interleave: "PIXEL" (GDAL default - one read returns every band of a region's tiles)
    or "BAND" (each band a contiguous plane, so reading one band/crop touches only it;
    use for multi-band COGs read one layer at a time). Irrelevant for single-band.
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
        return mem.read()
