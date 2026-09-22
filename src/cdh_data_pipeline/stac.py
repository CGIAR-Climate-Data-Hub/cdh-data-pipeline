"""Static STAC collections: read them, snapshot them as STAC GeoParquet.

Items come back as plain dicts so a recipe can normalise upstream metadata before
writing, and hand them to ``mosaic.write_vrt`` / ``write_gti`` unchanged.
"""

import asyncio
import logging
from urllib import parse

import rustac

from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import open_store

# rustac logs one INFO line per object read; that is per tile here.
logging.getLogger("stac_io").setLevel(logging.WARNING)

# urljoin only resolves relative hrefs for schemes it knows; teach it object stores.
for _scheme in ("s3", "gs", "az", "abfs"):
    parse.uses_relative.append(_scheme)
    parse.uses_netloc.append(_scheme)


def read_stac_collection(url):
    """Read a static STAC collection and its items, all links and hrefs made absolute.

    Credentials come from the environment; set
    ``AWS_SKIP_SIGNATURE=true`` for public buckets. Asset hrefs are resolved
    against ``url``, so pass HTTPS for public data if readers should need no setup.
    """
    return asyncio.run(_read(url))


async def _read(collection_url):
    collection = await rustac.read(collection_url)
    hrefs = [
        parse.urljoin(collection_url, link["href"])
        for link in collection["links"]
        if link["rel"] == "item"
    ]
    log.info("reading %d items from %s", len(hrefs), collection_url)
    items = await asyncio.gather(*(rustac.read(h) for h in hrefs))
    _absolutize(collection, collection_url)
    for item, href in zip(items, hrefs):
        _absolutize(item, href)
    return collection, items


def _absolutize(obj, href):
    """Resolve an item's or collection's links and assets against where it was read."""
    for link in obj["links"]:
        link["href"] = parse.urljoin(href, link["href"])
    for asset in obj.get("assets", {}).values():
        asset["href"] = parse.urljoin(href, asset["href"])


def write_stac_geoparquet(collection, items, url):
    """Write one collection's items as a STAC GeoParquet file at ``url``.

    rustac writes spec 1.0: WKB geometry, bbox covering column, collection JSON in
    the file metadata. STAC tools see a collection snapshot; GDAL 3.10+ with the
    Parquet driver also opens it as a mosaic via ``GTI:<url>``.
    """
    asyncio.run(_write_parquet(collection, items, url))


async def _write_parquet(collection, items, url):
    prefix, _, name = url.rpartition("/")
    store = open_store(prefix or ".")
    writer = await rustac.GeoparquetWriter.open(items, name, store=store)
    writer.add_collection(collection)
    await writer.finish()
    log.info("wrote %s (%d items)", url, len(items))
