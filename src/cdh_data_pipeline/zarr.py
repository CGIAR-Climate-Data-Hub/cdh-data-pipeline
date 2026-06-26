"""Zarr writing: float32-friendly compression + obstore-backed stores.

(``import zarr`` below is the installed package -- absolute imports, no shadow.)
"""

import rioxarray  # noqa: F401  registers .rio (CRS source for multiscale writes)
import xproj  # noqa: F401  registers .proj (topozarr reads the CRS via this convention)
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
    """blosc-zstd + byte-shuffle: ~10% smaller than plain zstd on float32."""
    return BloscCodec(
        cname="zstd", clevel=clevel, shuffle=BloscShuffle.shuffle, typesize=typesize
    )


def write_zarr(ds, url, encoding, *, consolidated=True):
    """Write a Dataset to an obstore-backed, GeoZarr-tagged zarr store.

    Each data variable is tagged with GeoZarr spatial:/proj: attrs derived from its
    rioxarray CRS + grid, so the store is recognized as geospatial. consolidated=True
    (default) bundles node metadata for single-request cloud opens (the ARCO
    convention); pass False for strict core-v3-spec portability (zarr-python warns
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
    ds, url, *, methods=None, factors=None, chunks_per_shard=None, compressors=None
):
    """Write ``ds`` as a multiscale (overview) GeoZarr store -- one group per variable.

    Like :func:`write_zarr`, but each variable also gets coarser overview levels, so the
    store serves full-resolution analysis and zoomed-out / web-map reads from one place.
    Layout: ``<store>.zarr/<var>/{0,1,2,...}/<var>`` (0 = native, then coarser). topozarr
    does the coarsening (native grid); we drive it once per variable -- which is how each
    variable keeps its own resampling method in one store -- and stitch a root group on top.

    methods : variable -> ``"mean"`` (default) | ``"sum"`` | ``"max"`` | ``"min"``.
        Per-variable: "sum" for totals (production, area), "mean" for densities/ratios.
    factors : cumulative downsample factors; native (1) is added if missing, e.g.
        ``[2, 4, 8]`` -> ``[1, 2, 4, 8]``. ``None`` -> topozarr picks a power-of-two ladder.
    chunks_per_shard : ``None`` (default) keeps tiles individually addressable; an int N
        wraps an N x N block of chunks into each zarr-v3 shard.
    compressors : ``None`` (default) -> topozarr's Rust write (fast, region-streamed, but
        the codec is fixed to zarr's zstd default). Pass a tuple of zarr-v3 codecs (e.g.
        ``(blosc_zstd(),)``) to control compression -- topozarr only coarsens and we write
        via xarray (no Rust kernel; build inline, not from a store read-back, for big data).
    """
    methods = methods or {}
    if factors is not None:
        factors = sorted({1, *factors})
    # topozarr reads the CRS via the proj convention (xproj); seed it from rioxarray's CRS.
    ds = ds.proj.assign_crs(spatial_ref=ds.rio.crs.to_string(), allow_override=True)
    root = ObjectStore(open_store(url))
    zarr.open_group(root, mode="w").attrs.update(ds.attrs)  # root group + dataset attrs
    for var in ds.data_vars:
        pyr = create_pyramid(
            ds[[var]],
            factors=factors,
            method=methods.get(var, "mean"),
            chunks_per_shard=chunks_per_shard,
        )
        sub = ObjectStore(open_store(f"{url}/{var}"))
        if compressors is None:
            pyr.write(sub, mode="a")  # topozarr Rust kernel; zstd-default codec
        else:  # topozarr coarsens, xarray writes -> we set the codec via the encoding
            enc = pyr.encoding
            for level in enc.values():
                for var_enc in level.values():
                    var_enc["compressors"] = compressors
            pyr.as_datatree().to_zarr(
                sub, mode="a", zarr_format=3, consolidated=False, encoding=enc
            )
    zarr.consolidate_metadata(root)
    print(f"wrote {url} ({len(ds.data_vars)} vars, multiscale)")
