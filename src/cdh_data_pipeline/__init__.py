"""cdh_data_pipeline -- shared plumbing for turning source rasters into ARCO zarr + COGs.

Recipes (see ``recipes/``) hold the per-dataset read + assembly logic and call
these helpers for the identical write side.
"""

from cdh_data_pipeline.cog import COG_OPTS, make_cog
from cdh_data_pipeline.recipe import run
from cdh_data_pipeline.storage import open_raster, open_store
from cdh_data_pipeline.zarr import blosc_zstd, write_multiscale_zarr, write_zarr

__all__ = [
    "COG_OPTS",
    "blosc_zstd",
    "make_cog",
    "open_raster",
    "open_store",
    "run",
    "write_multiscale_zarr",
    "write_zarr",
]
