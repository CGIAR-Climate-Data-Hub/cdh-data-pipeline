"""Zarr writers for geospatial raster datasets."""

from functools import partial
from dataclasses import dataclass

import numpy as np
import rioxarray  # noqa: F401  registers .rio
import xproj  # noqa: F401  registers .proj
import zarr
from topozarr import attach_geozarr_metadata, create_pyramid
from zarr.codecs import BloscCodec, BloscShuffle, ZstdCodec
from zarr.storage import ObjectStore

from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import clear_store, open_store

_SHAPE_KEYS = {"chunks", "shards"}

TIME_UNITS = "days since 1980-01-01"
EPOCH = np.datetime64("1980-01-01", "D")

@dataclass(frozen=True)
class CubeGeometry:
    """Chunk and shard geometry of the datacube.

    Chunks are the unit of decompression (and of a single HTTP range request
    once the store is hosted remotely). Shards are the unit of *file*: a shard
    bundles many chunks into one object so the store stays at a few thousand
    files instead of a few hundred thousand, which matters for object stores
    and for Hugging Face repos.

    Each shard extent must be an exact multiple of the corresponding chunk
    extent; :meth:`validate` enforces that.
    """

    time_chunk: int = 128
    lat_chunk: int = 100
    lon_chunk: int = 100
    time_shard: int = 128
    lat_shard: int = 600
    lon_shard: int = 600
    sharded: bool = False

    @property
    def chunks(self) -> tuple[int, int, int]:
        return (self.time_chunk, self.lat_chunk, self.lon_chunk)

    @property
    def shards(self) -> tuple[int, int, int] | None:
        if not self.sharded:
            return None
        return (self.time_shard, self.lat_shard, self.lon_shard)

    def validate(self, n_lat: int, n_lon: int) -> None:
        if not self.sharded:
            return
        for name, shard, chunk in (
            ("time", self.time_shard, self.time_chunk),
            ("lat", self.lat_shard, self.lat_chunk),
            ("lon", self.lon_shard, self.lon_chunk),
        ):
            if shard % chunk:
                raise ValueError(
                    f"{name} shard ({shard}) must be a whole multiple of the "
                    f"{name} chunk ({chunk})"
                )
        import warnings

        for name, size, shard in (("lat", n_lat, self.lat_shard),
                                  ("lon", n_lon, self.lon_shard)):
            if size % shard:
                warnings.warn(
                    f"grid {name} size ({size}) is not a whole number of "
                    f"{name} shards ({shard}); the trailing shard holds "
                    f"{size % shard} row(s)", stacklevel=2)

    def band_bytes(self) -> int:
        """Peak RAM for one (time_shard, lat_shard, full-width) staging block."""
        return self.time_shard * self.lat_shard * 7200 * 2



def compressors(codec: str = "blosc") -> list:
    """Compressor chain for a data array.

    ``blosc`` (zstd-5 + bitshuffle) is smallest for sparse, zero-heavy fields
    such as precipitation -- about 15% better than plain zstd on CHIRPS. For
    dense smooth fields (temperature) the two are within a few percent.

    ``zstd`` is plain Zstandard, which GDAL can read; the Blosc variant it
    generally cannot, since most GDAL builds ship without Blosc support. Choose
    ``zstd`` when the store needs to be readable outside the Python stack.
    """
    if codec == "zstd":
        return [ZstdCodec(level=5)]
    if codec == "blosc":
        return [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle)]
    raise ValueError(f"unknown codec {codec!r}; expected 'blosc' or 'zstd'")


def blosc_zstd(typesize=4, clevel=9, *, shuffle=False):
    """Blosc Zstd codec. For integer data, set ``shuffle`` and ``typesize``.

    ``shuffle`` is ``True`` for byte shuffle or ``"bitshuffle"``.
    """
    sh = shuffle if isinstance(shuffle, str) else "shuffle" if shuffle else "noshuffle"
    return BloscCodec(cname="zstd", clevel=clevel, shuffle=sh, typesize=typesize)


def _check_range(a, lo, hi, name):
    """Return ``a`` unless a non-NaN value lies outside lo..hi."""
    # fmin/fmax skip NaN (written as the fill value) and allocate no masks.
    amin = np.fmin.reduce(a, axis=None, initial=lo)
    amax = np.fmax.reduce(a, axis=None, initial=hi)
    if amin < lo or amax > hi:
        raise ValueError(
            f"{name} range {amin}..{amax} overflows its packed range {lo}..{hi}"
        )
    return a


def check_packable(ds, encoding):
    """Return ``ds`` with a range check on every float variable packed to integers.

    xarray applies ``scale_factor``/``add_offset`` and casts without checking, so
    out-of-range values silently wrap or become the fill value. The check runs per
    dask chunk as the data is written (or immediately for numpy data).
    """
    checked = {}
    for name, enc in (encoding or {}).items():
        if name not in ds or ds[name].dtype.kind != "f":
            continue
        dtype = np.dtype(enc.get("dtype", ds[name].dtype))
        if dtype.kind not in "iu":
            continue
        info, fill = np.iinfo(dtype), enc.get("_FillValue")
        lo, hi = info.min + (fill == info.min), info.max - (fill == info.max)
        scale, offset = enc.get("scale_factor", 1), enc.get("add_offset", 0)
        check = partial(
            _check_range, lo=lo * scale + offset, hi=hi * scale + offset, name=name
        )
        data = ds[name].data
        checked[name] = ds[name].copy(
            data=data.map_blocks(check) if hasattr(data, "dask") else check(data)
        )
    return ds.assign(checked)


def _vlen_str_coords(ds):
    """Cast unicode coords to object so zarr writes vlen strings."""
    str_coords = {c: ds[c].astype(object) for c in ds.coords if ds[c].dtype.kind == "U"}
    return ds.assign_coords(str_coords) if str_coords else ds


def _replace_sum_levels(dt, var, factors):
    """Recompute sum overviews so all-NaN windows stay NaN, not 0."""
    # Drop coords: xarray passes min_count to coord reducers, which rejects it.
    prev = dt["0"][var].drop_vars(dt["0"][var].coords)
    for lvl in range(1, len(factors)):
        step = factors[lvl] // factors[lvl - 1]
        coarse = prev.coarsen(x=step, y=step, boundary="trim")
        prev = coarse.sum(min_count=1)
        dt[str(lvl)][var] = prev.assign_attrs(dt[str(lvl)][var].attrs)


def _geozarr_attrs(da, crs):
    """GeoZarr proj + spatial attrs for one array."""
    # geozarr-toolkit 0.1.2 writes dead convention URLs; revisit once 0.1.3 ships.
    return attach_geozarr_metadata(da.to_dataset(), crs=crs).attrs

def open_for_write(store: str, **kwargs):
    """Open a group for modification with metadata read fresh, not consolidated.

    This matters more than it looks. ``zarr.consolidate_metadata`` caches every
    array's metadata in the root ``zarr.json``, and ``open_group`` serves array
    handles from that cache by default. After an array is resized, the cache
    still carries the old shape, so a *re-fetched* handle reports the old
    length and any write past it is **silently clipped** -- no exception, no
    warning, the data simply never lands.

    Bypassing the cache while writing costs one metadata read per array and
    removes the failure mode entirely. Re-consolidate once the writes are done.
    """
    import zarr

    return zarr.open_group(store, mode="r+", use_consolidated=False, **kwargs)



def _open_zarr_store(url):
    """Open a .zarr store, deleting anything already there."""
    if not url.rstrip("/").endswith(".zarr"):
        raise ValueError(f"refusing to overwrite non-.zarr store: {url}")
    store = open_store(url)
    clear_store(store)
    return store


def write_zarr(ds, url, encoding=None, *, consolidated=True):
    """Write a Dataset as a GeoZarr v3 store.

    ``encoding`` maps variable name to zarr encoding, e.g.
    ``{"var": {"chunks": (1080, 1080), "compressors": (blosc_zstd(),)}}``.
    Variables packed to integers are range-checked (see ``check_packable``).
    """
    store = _open_zarr_store(url)
    ds = check_packable(_vlen_str_coords(ds), encoding)
    crs = ds.rio.crs.to_string()
    for da in ds.data_vars.values():
        da.attrs.update(_geozarr_attrs(da, crs))
    log.info("writing %s (%d vars)", url, len(ds.data_vars))
    ds.to_zarr(
        ObjectStore(store),
        mode="w",
        zarr_format=3,
        consolidated=consolidated,
        encoding=encoding,
    )
    log.info("wrote %s", url)


def _write_pyramid(pyr, target, variables, methods, encoding, level_fn, crs):
    """Write one pyramid to ``target`` and stamp CRS attrs on each array."""
    dt = pyr.as_datatree()
    sum_vars = [v for v in variables if methods.get(v, "mean") == "sum"]
    # topozarr's fast writer can't do custom encoding or NaN-preserving sums.
    if encoding is not None or level_fn is not None or sum_vars:
        for v in sum_vars:
            _replace_sum_levels(dt, v, pyr.factors)
        enc = {}
        for k in pyr.encoding:
            name = k.strip("/")
            sizes = dict(dt[name].sizes)
            enc[k] = {}
            for v in variables:
                shapes = pyr.encoding[k][v]
                if level_fn:
                    custom = level_fn(v, int(name), sizes)
                    if not isinstance(custom, dict) or custom.keys() - _SHAPE_KEYS:
                        raise ValueError(
                            f"chunking must return chunks/shards; got {custom!r}"
                        )
                    shapes = {**shapes, **custom}
                enc[k][v] = {**(encoding or {}).get(v, {}), **shapes}
        dt.to_zarr(target, mode="a", zarr_format=3, consolidated=False, encoding=enc)
    else:
        pyr.write(target, mode="a")
    # GDAL reads the CRS from the array; topozarr only writes it on the group.
    grp = zarr.open_group(target, mode="r+")
    for k in pyr.encoding:
        name = k.strip("/")
        for v in variables:
            grp[f"{name}/{v}"].attrs.update(_geozarr_attrs(dt[name][v], crs))


def write_multiscale_zarr(
    ds,
    url,
    *,
    factors,
    methods=None,
    encoding=None,
    chunking=None,
    layout="variable",
):
    """Write a multiscale GeoZarr store. Level 0 is native resolution.

    ``layout="variable"`` writes ``<var>/<level>/<var>``, one pyramid per variable.
    ``layout="level"`` writes ``<level>/<var>`` and needs one method for all variables.

    ``factors``: cumulative downsampling factors, e.g. ``[2, 4, 8]``.
    ``methods``: variable -> ``"mean"`` (default), ``"sum"``, ``"max"``, ``"min"``
    or ``"nearest"`` (use for categorical data).
    ``encoding``: variable -> zarr encoding, applied to every level.
    ``chunking``: int chunks per shard, or a ``(var, level, sizes)`` callable
    returning ``{"chunks": ..., "shards": ...}``. ``None`` writes unsharded.
    """
    methods = methods or {}
    if layout not in ("variable", "level"):
        raise ValueError(f"layout must be 'variable' or 'level'; got {layout!r}")
    used_methods = {methods.get(v, "mean") for v in ds.data_vars}
    if layout == "level" and len(used_methods) > 1:
        raise ValueError(f"layout='level' needs one method for all vars; got {methods}")
    factors = sorted({1, *factors})
    per_shard = chunking if isinstance(chunking, int) else None
    level_fn = chunking if callable(chunking) else None
    if chunking is not None and per_shard is None and level_fn is None:
        raise TypeError(f"chunking must be an int or callable; got {chunking!r}")
    ds = _vlen_str_coords(ds)
    # topozarr reads CRS from xproj metadata.
    crs = ds.rio.crs.to_string()
    ds = ds.proj.assign_crs(spatial_ref=crs, allow_override=True)
    store = _open_zarr_store(url)
    root = ObjectStore(store)
    zarr.open_group(root, mode="w")
    variables = list(ds.data_vars)
    log.info("writing %s (%d vars, multiscale, %s-first)", url, len(variables), layout)
    if layout == "level":
        pyr = create_pyramid(
            ds,
            factors=factors,
            method=next(iter(used_methods)),
            chunks_per_shard=per_shard,
        )
        _write_pyramid(pyr, root, variables, methods, encoding, level_fn, crs)
    else:
        for var in variables:
            log.info("  pyramid %s", var)
            pyr = create_pyramid(
                ds[[var]],
                factors=factors,
                method=methods.get(var, "mean"),
                chunks_per_shard=per_shard,
            )
            sub = ObjectStore(open_store(f"{url}/{var}"))
            _write_pyramid(pyr, sub, [var], methods, encoding, level_fn, crs)
    # Merge, don't replace: level layout already put multiscales attrs on the root.
    zarr.open_group(root, mode="r+").attrs.update(ds.attrs)
    zarr.consolidate_metadata(root)
    log.info("wrote %s", url)
