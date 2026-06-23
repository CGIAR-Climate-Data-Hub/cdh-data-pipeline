"""Object store access + source raster reading.

URLs pick the backend by scheme, so a local dir swaps for a cloud bucket later
with no code change.
"""

import rioxarray as rxr
import xarray as xr
from obstore.store import LocalStore, from_url


def open_store(url):
    """obstore store for a base URL/path (local path -> LocalStore, else by scheme)."""
    return from_url(url) if "://" in url else LocalStore(url, mkdir=True)


def open_raster(url, name=None, *, chunks=None):
    """Read a single-band raster as a clean float32 DataArray (nodata -> NaN).

    chunks controls loading -- a memory-vs-simplicity trade-off:
      - None -> eager: read the whole array into memory now. Simplest; use when a
        recipe opens only a handful of small rasters.
      - -1   -> lazy: one dask chunk for the whole grid, not read until computed. Use
        when stacking many layers into one dataset so they stream through to_zarr
        instead of all loading into RAM at once (e.g. mapspam's 552 layers).
    Per-file STATISTICS_* attrs are dropped; the caller sets clean attrs on the variable.
    """
    da = rxr.open_rasterio(url, masked=True, chunks=chunks)
    assert isinstance(da, xr.DataArray)  # single-band raster -> DataArray
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    da = da.astype("float32")
    da.attrs.clear()
    return da.rename(name) if name else da
