"""Zarr writing: float32-friendly compression + obstore-backed stores.

(``import zarr`` below is the installed package -- absolute imports, no shadow.)
"""

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

from cdh_data_pipeline.storage import open_store


def blosc_zstd(typesize=4, clevel=9):
    """blosc-zstd + byte-shuffle default"""
    return BloscCodec(
        cname="zstd", clevel=clevel, shuffle=BloscShuffle.shuffle, typesize=typesize
    )


def write_zarr(ds, url, encoding, *, consolidated=True):
    """Write a Dataset to an obstore-backed, GeoZarr-tagged zarr store.

    Each data variable is tagged with GeoZarr attrs derived from its
    rioxarray attrs, so the store is recognized as geospatial. consolidated=True
    (default) bundles node metadata for single-request cloud opens;
    pass False for strict core-v3-spec portability (zarr-python warns
    it's an extension, not in the v3 spec).
    """
    # zarr v3 has no stable spec for fixed-length unicode; store string coords as
    # variable-length (object dtype) so the store stays portable across zarr libraries.
    str_coords = {c: ds[c].astype(object) for c in ds.coords if ds[c].dtype.kind == "U"}
    if str_coords:
        ds = ds.assign_coords(str_coords)

    conventions = create_zarr_conventions(
        SpatialConventionMetadata(), ProjConventionMetadata()
    )
    for da in ds.data_vars.values():
        da.attrs.update(from_rioxarray(da), zarr_conventions=conventions)
    ds.to_zarr(
        ObjectStore(open_store(url)),
        mode="w",
        zarr_format=3,
        consolidated=consolidated,
        encoding=encoding,
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
):
    """Write a multiscale GeoZarr store (zarr w/ internal pyramids/overviews).

    The output layout is ``<store>.zarr/<var>/{0,1,2,...}/<var>``. Level 0 contains
    the native-resolution data; subsequent levels contain coarsened overviews.

    ``methods`` maps variable names to downsampling methods: ``"mean"``, ``"sum"``,
    ``"max"``, or ``"min"``. Unspecified variables use ``"mean"``.

    ``factors`` gives cumulative downsampling factors. Factor 1 is added
    automatically if missing. If omitted, topozarr chooses a power-of-two pyramid.

    ``compressors`` is a shortcut for applying the same Zarr codec tuple to all
    variables. It is merged into ``encoding``; explicit per-variable ``encoding``
    entries take precedence.

    ``encoding`` is a mapping of variable name to Zarr encoding and is applied to
    every level of that variable. Use it for settings that should not vary by
    overview level, such as dtype, scale factors, fill values, and compressors.

    ``chunking`` may be either an integer or a callable. An integer is passed to
    topozarr as ``chunks_per_shard`` and uses the fast Rust writer, unless
    ``encoding`` or ``compressors`` require the custom xarray path. A callable must
    accept ``(var, level_index, sizes)`` and return a dict containing ``"chunks"``
    and/or ``"shards"`` for that level.
    """
    methods = methods or {}
    if factors is not None:
        factors = sorted({1, *factors})
    # chunking is int (N chunks/shard, fast Rust path) XOR callable (per-level, xarray path)
    per_shard = chunking if isinstance(chunking, int) else None
    level_fn = chunking if callable(chunking) else None
    if chunking is not None and per_shard is None and level_fn is None:
        raise TypeError(
            "chunking must be an int (N chunks/shard) or a callable "
            f"(var, level, sizes) -> {{'chunks':..,'shards':..}}; got {type(chunking).__name__}"
        )
    # `compressors` = shorthand for that codec on every variable -> fold into encoding
    if compressors is not None:
        encoding = {
            v: {"compressors": compressors, **(encoding or {}).get(v, {})}
            for v in ds.data_vars
        }
    # topozarr reads the CRS via the proj convention (xproj); seed it from rioxarray's CRS.
    ds = ds.proj.assign_crs(spatial_ref=ds.rio.crs.to_string(), allow_override=True)
    root = ObjectStore(open_store(url))
    zarr.open_group(root, mode="w").attrs.update(ds.attrs)  # root group + dataset attrs
    for var in ds.data_vars:
        pyr = create_pyramid(
            ds[[var]],
            factors=factors,
            method=methods.get(var, "mean"),
            chunks_per_shard=per_shard,
        )
        sub = ObjectStore(open_store(f"{url}/{var}"))
        # custom layout (per-var encoding and/or per-level chunks) -> xarray write
        if encoding is not None or level_fn is not None:
            dt = pyr.as_datatree()
            base = (encoding or {}).get(var, {})
            enc = {}
            for k in pyr.encoding:  # keys "/0","/1",... (0 = native)
                lvl, sizes = int(k.strip("/")), dict(dt[k.strip("/")].sizes)
                shapes = level_fn(var, lvl, sizes) if level_fn else {}
                if not isinstance(shapes, dict) or shapes.keys() - {"chunks", "shards"}:
                    raise ValueError(
                        f"chunking({var!r}, {lvl}, ...) must return a dict of "
                        f"'chunks'/'shards' (encoding goes in `encoding=`); got {shapes!r}"
                    )
                enc[k] = {var: {**base, **shapes}}
            dt.to_zarr(sub, mode="a", zarr_format=3, consolidated=False, encoding=enc)
        else:
            pyr.write(sub, mode="a")  # topozarr Rust kernel; zstd-default codec
    zarr.consolidate_metadata(root)
    print(f"wrote {url} ({len(ds.data_vars)} vars, multiscale)")
