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

    chunks=None loads eager; chunks=-1 stays lazy as one dask chunk. Per-file
    STATISTICS_* attrs are dropped; the caller sets clean attrs on the variable.
    """
    da = rxr.open_rasterio(url, masked=True, chunks=chunks)
    assert isinstance(da, xr.DataArray)  # single-band raster -> DataArray
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    da = da.astype("float32")
    da.attrs.clear()
    return da.rename(name) if name else da
