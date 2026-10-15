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
    """Return an fsspec filesystem for a local path or URL, for pyarrow writers."""
    return FsspecStore(url.split("://")[0] if "://" in url else "file")


def clear_store(store):
    """Delete all keys from an obstore store."""
    for batch in store.list():
        paths = [meta["path"] for meta in batch]
        if paths:
            store.delete(paths)


# S3 stores whatever Content-Type the uploader sends, and obstore sends none.
# VRTs get "attachment" so browsers download them instead of rendering the XML.
_ATTRS = {
    ".json": {"Content-Type": "application/json"},
    ".vrt": {"Content-Type": "application/xml", "Content-Disposition": "attachment"},
    ".gpkg": {"Content-Type": "application/geopackage+sqlite3"},
    ".tif": {
        "Content-Type": "image/tiff; application=geotiff; profile=cloud-optimized"
    },
}


def put_bytes(url, data):
    """Write bytes to a local path or object-store URL, with a Content-Type by suffix."""
    prefix, _, name = url.rpartition("/")
    # LocalStore raises NotImplementedError on any attributes
    attrs = _ATTRS.get(Path(name).suffix) if "://" in url else None
    open_store(prefix or ".").put(name, data, attributes=attrs)


def write_json(url, data):
    """Write a dict as JSON to a local path or object-store URL."""
    put_bytes(url, json.dumps(data, indent=2).encode())
    log.info("wrote %s", url)


def put_file(url, path):
    """Upload a local file to a local path or object-store URL."""
    put_bytes(url, Path(path).read_bytes())


def open_raster(url, name=None, *, chunks=None):
    """Read a raster as float32, nodata as NaN, attrs cleared.

    Single-band rasters drop the ``band`` dim; multi-band keep it (numbered 1..N).
    ``chunks=None`` loads eagerly; ``chunks=-1`` keeps it lazy as one dask chunk.
    """
    da = rxr.open_rasterio(url, masked=True, chunks=chunks)
    assert isinstance(da, xr.DataArray)
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    da = da.astype("float32")
    da.attrs.clear()
    return da.rename(name) if name else da
