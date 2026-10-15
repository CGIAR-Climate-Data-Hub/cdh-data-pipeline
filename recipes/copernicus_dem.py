"""Copernicus DEM GLO-30 / GLO-90 -> STAC GeoParquet + GTI mosaics.

Indexes the AWS Open Data COGs in place; nothing is copied (GLO-30 alone is
over a terabyte). Writes one GeoParquet snapshot of the Earth Search items per
product, plus a GTI tile index that GDAL 3.8+ opens as one global raster:
``GTI:<url>/glo-90.gti.gpkg``.

Items come from Earth Search rather than the upstream static STAC because they
carry proj:shape/transform, raster:bands and a real datetime. Their hrefs are
``s3://``; the bucket is public, so they are rewritten to HTTPS to keep readers
credential-free.

Tiles are 1x1 degree with no nodata; the ocean has no tiles and reads as 0.
Pixel width grows with latitude (1x at the equator to 10x above 85N/S), so the
index is at the equatorial resolution and GTI resamples the wider tiles on read
(nearest neighbour, so columns are duplicated above 50N/S).
GLO-30 is missing tiles for a few countries not yet released by Copernicus.

Run from the repo root: uv run recipes/copernicus_dem.py
"""

import rustac

from cdh_data_pipeline import run, write_gti, write_stac_geoparquet

API = "https://earth-search.aws.element84.com/v1"
OUTPUT = "s3://digital-atlas/cdh/data/copernicus-dem-2021"
PRODUCTS = {"glo-30": "cop-dem-glo-30", "glo-90": "cop-dem-glo-90"}


def snapshot(name, collection_id):
    collection = rustac.read_sync(f"{API}/collections/{collection_id}")
    items = rustac.search_sync(API, collections=collection_id, limit=1000)
    for item in items:  # swap s3 prefix for more user friendly https
        href = item["assets"]["data"]["href"]
        bucket, _, key = href.removeprefix("s3://").partition("/")
        item["assets"]["data"]["href"] = f"https://{bucket}.s3.amazonaws.com/{key}"
    # write_gti takes resolution/overviews from the first tile: make it an equatorial one
    items.sort(key=lambda item: abs(item["bbox"][1]))
    write_stac_geoparquet(collection, items, f"{OUTPUT}/{name}.parquet")
    write_gti(f"{OUTPUT}/{name}.gti.gpkg", items)


def build():
    for name, collection_id in PRODUCTS.items():
        snapshot(name, collection_id)


if __name__ == "__main__":
    run(build)
