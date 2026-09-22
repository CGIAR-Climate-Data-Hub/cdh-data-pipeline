"""JRC GloFAS global flood hazard maps -> STAC GeoParquet + VRT mosaics.

The source is already tiled COGs with a static STAC catalog on source.coop. We do
not copy the data. Each collection becomes a STAC GeoParquet snapshot, and the
tiles are exposed as VRT mosaics with return period as the band axis:
``<OUTPUT>/depth.vrt`` (7 bands, rp10..rp500) and ``<OUTPUT>/hazard.vrt``.

``depth-*`` is flood depth in metres (float32). ``hazard-*`` is the publisher's
three-class reclassification of depth (uint8); the class breaks are not documented.

INPUT is the HTTPS mirror on purpose: asset hrefs are written as read from the
collection, so HTTPS gives readers ``/vsicurl`` paths that need no AWS setup.

Run from the repo root: uv run recipes/jrc_glofas.py
"""

from cdh_data_pipeline import (
    read_stac_collection,
    run,
    write_stac_geoparquet,
    write_vrt,
)

INPUT = "https://data.source.coop/nlebovits/jrc-glofas"
OUTPUT = "s3://digital-atlas/cdh/data/jrc-glofas-v2.1.2"

RETURN_PERIODS = (10, 20, 50, 75, 100, 200, 500)
STACKED = ("depth", "hazard")  # one VRT each, band per return period
SINGLE = ("permanent-water", "spurious-depths")  # auxiliary masks, one VRT each


def fix_transform(item):
    """Upstream bug: proj:transform is in GDAL order [x0, dx, 0, y0, 0, dy].

    STAC (and GDAL's STACIT driver) want affine order [dx, 0, x0, 0, dy, y0].
    """
    t = item["properties"]["proj:transform"]
    if t[2] == 0 and t[4] == 0 and t[1] != 0:
        item["properties"]["proj:transform"] = [t[1], t[2], t[0], t[4], t[5], t[3]]
    return item


def snapshot(name):
    collection, items = read_stac_collection(f"{INPUT}/{name}/collection.json")
    items = [fix_transform(item) for item in items]
    write_stac_geoparquet(collection, items, f"{OUTPUT}/{name}.parquet")
    return items


def build():
    for kind in STACKED:
        bands = {f"rp{rp}": snapshot(f"{kind}-rp{rp}") for rp in RETURN_PERIODS}
        write_vrt(f"{OUTPUT}/{kind}.vrt", bands)
    for name in SINGLE:
        write_vrt(f"{OUTPUT}/{name}.vrt", {name: snapshot(name)})


if __name__ == "__main__":
    run(build)
