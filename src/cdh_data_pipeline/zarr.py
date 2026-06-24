"""Zarr writing: float32-friendly compression + obstore-backed stores.

(``import zarr`` below is the installed package -- absolute imports, no shadow.)
"""

from geozarr_toolkit import (
    ProjConventionMetadata,
    SpatialConventionMetadata,
    create_zarr_conventions,
    from_rioxarray,
)
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
    bundles node metadata for single-request cloud opens (the ARCO convention);
    zarr-python warns it's not in the v3 spec -- expected; pass False for strict
    cross-implementation portability.
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
