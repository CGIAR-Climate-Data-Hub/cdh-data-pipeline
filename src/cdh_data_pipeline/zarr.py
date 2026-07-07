"""Zarr writers for geospatial raster datasets."""

import rioxarray  # noqa: F401  registers .rio
import xproj  # noqa: F401  registers .proj
import zarr
from geozarr_toolkit import (
    ProjConventionMetadata,
    SpatialConventionMetadata,
    create_zarr_conventions,
    from_rioxarray,
)
from topozarr import create_pyramid
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ObjectStore

from cdh_data_pipeline.storage import clear_store, open_store


def blosc_zstd(typesize=4, clevel=9, *, shuffle=False):
    """Return a Blosc Zstd codec.

    Leave shuffle off for noisy float32 rasters. Enable it for low-entropy integer data
    and set ``typesize`` to the dtype itemsize.
    """
    sh = BloscShuffle.shuffle if shuffle else BloscShuffle.noshuffle
    return BloscCodec(cname="zstd", clevel=clevel, shuffle=sh, typesize=typesize)


def _vlen_str_coords(ds):
    """Store unicode coordinate labels as portable variable-length strings."""
    str_coords = {c: ds[c].astype(object) for c in ds.coords if ds[c].dtype.kind == "U"}
    return ds.assign_coords(str_coords) if str_coords else ds


def _replace_sum_levels(dt, var, factors):
    """Recompute sum overviews so empty windows stay missing."""
    # Avoid reducing coordinates; xarray forwards min_count to coord reducers too.
    prev = dt["0"][var].drop_vars(dt["0"][var].coords)
    for lvl in range(1, len(factors)):
        step = factors[lvl] // factors[lvl - 1]
        coarse = prev.coarsen(x=step, y=step, boundary="trim")
        prev = coarse.sum(min_count=1)
        dt[str(lvl)][var] = prev.assign_attrs(dt[str(lvl)][var].attrs)


def _open_zarr_store(url):
    """Open a zarr store after clearing an existing .zarr prefix."""
    if not url.rstrip("/").endswith(".zarr"):
        raise ValueError(f"refusing to overwrite non-.zarr store: {url}")
    store = open_store(url)
    clear_store(store)
    return store


def write_zarr(ds, url, encoding, *, consolidated=True):
    """Write a Dataset to an obstore-backed GeoZarr store.

    Each data variable receives GeoZarr attrs from rioxarray. Consolidated metadata
    makes cloud opens cheaper, but it is a zarr-python extension for v3 stores.
    """
    store = _open_zarr_store(url)
    ds = _vlen_str_coords(ds)

    conventions = create_zarr_conventions(
        SpatialConventionMetadata(), ProjConventionMetadata()
    )
    for da in ds.data_vars.values():
        da.attrs.update(from_rioxarray(da), zarr_conventions=conventions)
    ds.to_zarr(
        ObjectStore(store),
        mode="w",
        zarr_format=3,
        consolidated=consolidated,
        encoding=encoding,
    )
    print(f"wrote {url} ({len(ds.data_vars)} vars)")


def _write_pyramid(
    pyr, target, variables, methods, encoding, level_fn, conventions, url
):
    """Write one (possibly multi-variable) pyramid to ``target`` and stamp CRS attrs.

    Shared by both layouts: variable-first calls it once per variable into a
    ``<var>`` subgroup; level-first calls it once for the whole dataset into the root.
    """
    dt = pyr.as_datatree()
    sum_vars = [v for v in variables if methods.get(v, "mean") == "sum"]
    # Use xarray when we need custom encoding or sum's missing-data semantics.
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
                    if not isinstance(custom, dict) or custom.keys() - {
                        "chunks",
                        "shards",
                    }:
                        raise ValueError(
                            f"chunking({v!r}, {name}, ...) must return a dict of "
                            f"'chunks'/'shards' (encoding goes in `encoding=`); got {custom!r}"
                        )
                    shapes = {**shapes, **custom}
                enc[k][v] = {**(encoding or {}).get(v, {}), **shapes}
        dt.to_zarr(target, mode="a", zarr_format=3, consolidated=False, encoding=enc)
    else:
        # topozarr's Rust writer only supports S3 obstore targets.
        io = "rust" if url.startswith("s3://") else "python"
        pyr.write(target, mode="a", io=io)
    # GDAL/QGIS resolve the CRS from proj attrs on the array node itself;
    # topozarr only writes them on the variable group.
    grp = zarr.open_group(target, mode="r+")
    for k in pyr.encoding:
        name = k.strip("/")
        for v in variables:
            grp[f"{name}/{v}"].attrs.update(
                from_rioxarray(dt[name][v]), zarr_conventions=conventions
            )


def write_multiscale_zarr(
    ds,
    url,
    *,
    methods=None,
    factors=None,
    compressors=None,
    encoding=None,
    chunking=None,
    layout="variable",
):
    """Write a multiscale GeoZarr store.

    ``layout`` picks the store shape. ``"variable"`` (default) writes
    ``<store>.zarr/<var>/{0,1,2,...}/<var>`` — each variable is its own pyramid, so
    each can use its own downsampling method. ``"level"`` writes
    ``<store>.zarr/{0,1,2,...}/<var>`` — one pyramid for the whole dataset, so
    ``xr.open_zarr(group="0")`` yields every variable at that resolution. Level 0 is
    native resolution; higher levels are coarsened overviews.

    ``"level"`` requires a single shared method for all variables: the multiscales
    convention records one ``resampling_method`` per group, so it cannot express
    per-variable methods. Mixed ``methods`` with ``layout="level"`` raises.

    ``methods`` maps variable names to downsampling methods: ``"mean"``, ``"sum"``,
    ``"max"``, or ``"min"``. Unspecified variables use ``"mean"``. ``"sum"`` variables
    use ``min_count=1`` so all-missing windows stay missing.

    ``factors`` gives cumulative downsampling factors. Factor 1 is added
    automatically if missing. If omitted, topozarr chooses a power-of-two pyramid.

    ``compressors`` is a shortcut for applying the same Zarr codec tuple to all
    variables. It is merged into ``encoding``; explicit per-variable ``encoding``
    entries take precedence.

    ``encoding`` is a mapping of variable name to Zarr encoding and is applied to
    every level of that variable. Use it for settings that should not vary by
    overview level, such as dtype, scale factors, fill values, and compressors.

    ``chunking`` may be either an integer or a callable. Integers are passed to
    topozarr as ``chunks_per_shard``. Callables receive ``(var, level_index, sizes)``
    and return ``"chunks"`` and/or ``"shards"`` overrides for that level.
    """
    methods = methods or {}
    if layout not in ("variable", "level"):
        raise ValueError(f"layout must be 'variable' or 'level'; got {layout!r}")
    used_methods = {methods.get(v, "mean") for v in ds.data_vars}
    if layout == "level" and len(used_methods) > 1:
        raise ValueError(
            "layout='level' needs one shared method for all variables (multiscales "
            f"records one resampling_method per group); got {methods}. "
            "Use layout='variable' for per-variable methods."
        )
    if factors is not None:
        factors = sorted({1, *factors})
    # Integer chunking stays on topozarr's writer; callables need explicit encoding.
    per_shard = chunking if isinstance(chunking, int) else None
    level_fn = chunking if callable(chunking) else None
    if chunking is not None and per_shard is None and level_fn is None:
        raise TypeError(
            "chunking must be an int (N chunks/shard) or a callable "
            f"(var, level, sizes) -> {{'chunks':..,'shards':..}}; got {type(chunking).__name__}"
        )
    # Shared codec shortcut; variable-specific encoding still wins.
    if compressors is not None:
        encoding = {
            v: {"compressors": compressors, **(encoding or {}).get(v, {})}
            for v in ds.data_vars
        }
    ds = _vlen_str_coords(ds)
    # topozarr reads CRS from xproj metadata.
    ds = ds.proj.assign_crs(spatial_ref=ds.rio.crs.to_string(), allow_override=True)
    store = _open_zarr_store(url)
    root = ObjectStore(store)
    zarr.open_group(root, mode="w")
    conventions = create_zarr_conventions(
        SpatialConventionMetadata(), ProjConventionMetadata()
    )
    variables = list(ds.data_vars)
    if layout == "level":
        pyr = create_pyramid(
            ds,
            factors=factors,
            method=next(iter(used_methods)),
            chunks_per_shard=per_shard,
        )
        _write_pyramid(
            pyr, root, variables, methods, encoding, level_fn, conventions, url
        )
    else:
        for var in variables:
            pyr = create_pyramid(
                ds[[var]],
                factors=factors,
                method=methods.get(var, "mean"),
                chunks_per_shard=per_shard,
            )
            sub = ObjectStore(open_store(f"{url}/{var}"))
            _write_pyramid(
                pyr, sub, [var], methods, encoding, level_fn, conventions, url
            )
    # Root attrs last: level-first's pyramid write populates root.attrs
    # (multiscales); merging keeps it while adding the dataset attrs.
    zarr.open_group(root, mode="r+").attrs.update(ds.attrs)
    zarr.consolidate_metadata(root)
    print(f"wrote {url} ({len(variables)} vars, multiscale, {layout}-first)")
