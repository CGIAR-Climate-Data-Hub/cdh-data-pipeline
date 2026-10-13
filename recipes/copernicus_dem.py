"""Copernicus DEM GLO-30 / GLO-90 -> STAC GeoParquet + GTI mosaics.

Indexes the AWS Open Data COGs in place; nothing is copied (GLO-30 alone is
over a terabyte). Writes one GeoParquet snapshot of the upstream static STAC
per product, plus a GTI tile index that GDAL 3.8+ opens as one global raster:
``GTI:<url>/glo-90.gti.gpkg``.

Tiles are 1x1 degree with no nodata; the ocean has no tiles and reads as 0.
Pixel width grows with latitude (1x at the equator to 10x above 85N/S), so the
index is at the equatorial resolution and GTI resamples the wider tiles on read.
GLO-30 is missing tiles for a few countries not yet released by Copernicus.

The STAC buckets are public but have no item links, so items are listed from
the bucket rather than followed from the collection.

Run from the repo root: uv run recipes/copernicus_dem.py
"""

import asyncio
import json

from obstore.store import from_url

from cdh_data_pipeline import run, write_gti, write_stac_geoparquet

OUTPUT = "s3://digital-atlas/cdh/data/copernicus-dem-2021"
PRODUCTS = {"glo-30": "dem_cop_30", "glo-90": "dem_cop_90"}  # name -> collection id


def snapshot(name, collection_id):
    store = from_url(
        f"s3://copernicus-dem-{name[-2:]}m-stac",
        skip_signature=True,
        region="eu-central-1",
    )
    collection = json.loads(bytes(store.get(f"{collection_id}.json").bytes()))
    keys = [meta["path"] for batch in store.list("items") for meta in batch]
    items = asyncio.run(_read_items(store, keys))
    # write_gti takes resolution/overviews from the first tile: make it an equatorial one
    items.sort(key=lambda item: abs(item["bbox"][1]))
    write_stac_geoparquet(collection, items, f"{OUTPUT}/{name}.parquet")
    write_gti(f"{OUTPUT}/{name}.gti.gpkg", items, asset="elevation")


async def _read_items(store, keys):
    bodies = await asyncio.gather(*(store.get_async(k) for k in keys))
    return [json.loads(bytes(await body.bytes_async())) for body in bodies]


def build():
    for name, collection_id in PRODUCTS.items():
        snapshot(name, collection_id)


if __name__ == "__main__":
    run(build)
