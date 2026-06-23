"""Zarr writing: float32-friendly compression + obstore-backed stores.

(``import zarr`` below is the installed package -- absolute imports, no shadow.)
"""

from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ObjectStore

from cdh_data_pipeline.storage import open_store


def blosc_zstd(typesize=4, clevel=9):
    """blosc-zstd + byte-shuffle: ~10% smaller than plain zstd on float32."""
    return BloscCodec(
        cname="zstd", clevel=clevel, shuffle=BloscShuffle.shuffle, typesize=typesize
    )


def write_zarr(ds, url, encoding, *, consolidated=True):
    """Write a Dataset to an obstore-backed zarr store.

    consolidated=True bundles node metadata for single-request cloud opens (the ARCO
    convention). zarr-python warns it's not in the v3 spec -- expected; pass False for
    strict cross-implementation portability.
    """
    ds.to_zarr(
        ObjectStore(open_store(url)),
        mode="w",
        zarr_format=3,
        consolidated=consolidated,
        encoding=encoding,
    )
