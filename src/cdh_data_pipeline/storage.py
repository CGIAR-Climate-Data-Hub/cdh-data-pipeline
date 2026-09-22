"""Object-store access and raster loading."""

import json
from pathlib import Path

import rioxarray as rxr
import xarray as xr
from obstore.fsspec import FsspecStore
from obstore.store import LocalStore, from_url

from cdh_data_pipeline.recipe import log


def open_store(url):
    """Return an obstore store for a local path or URL."""
    return from_url(url) if "://" in url else LocalStore(url, mkdir=True)


def open_fs(url):
    """Return an fsspec filesystem for a URL, or None for a local path.

    Same obstore backend and env-var credentials as open_store; pyarrow writers
    want a filesystem rather than a store, and None lets them handle local paths.
    """
    return FsspecStore(url.split("://")[0]) if "://" in url else None


def clear_store(store):
    """Delete all keys from an obstore store."""
    for batch in store.list():
        paths = [meta["path"] for meta in batch]
        if paths:
            store.delete(paths)


def write_json(url, data):
    """Write a dict as JSON to a local path or object-store URL."""
    prefix, _, name = url.rpartition("/")
    open_store(prefix or ".").put(name, json.dumps(data, indent=2).encode())
    log.info("wrote %s", url)


def put_file(url, path):
    """Upload a local file to a local path or object-store URL."""
    prefix, _, name = url.rpartition("/")
    open_store(prefix or ".").put(name, Path(path).read_bytes())


def open_raster(url, name=None, *, chunks=None):
    """Read a single-band raster as float32 with nodata mapped to NaN.

    ``chunks=None`` loads eagerly. ``chunks=-1`` keeps one lazy Dask chunk per raster,
    useful when a recipe stacks many layers before writing.
    Source attrs are cleared; recipes set normalized metadata.
    """
    da = rxr.open_rasterio(url, masked=True, chunks=chunks)
    assert isinstance(da, xr.DataArray)
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    da = da.astype("float32")
    da.attrs.clear()
    return da.rename(name) if name else da
