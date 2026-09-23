"""Read static STAC collections and write them as STAC GeoParquet."""

import asyncio
import logging
from urllib import parse

import rustac

from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import open_store

# rustac logs an INFO line per item read
logging.getLogger("stac_io").setLevel(logging.WARNING)

# let urljoin resolve relative hrefs under object-store URLs
for _scheme in ("s3", "gs", "az", "abfs"):
    parse.uses_relative.append(_scheme)
    parse.uses_netloc.append(_scheme)


def read_stac_collection(url):
    """Return ``(collection, items)`` as dicts, with all hrefs made absolute.

    Hrefs resolve against ``url``, so use an HTTPS URL for public data to keep
    them credential-free. Set ``AWS_SKIP_SIGNATURE=true`` for public S3 buckets.
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
    """Resolve link and asset hrefs against ``href``."""
    for link in obj["links"]:
        link["href"] = parse.urljoin(href, link["href"])
    for asset in obj.get("assets", {}).values():
        asset["href"] = parse.urljoin(href, asset["href"])


def write_stac_geoparquet(collection, items, url):
    """Write items as STAC GeoParquet at ``url``, collection JSON in the metadata.

    GDAL 3.10+ can also open it as a mosaic via ``GTI:<url>``.
    """
    asyncio.run(_write_parquet(collection, items, url))


async def _write_parquet(collection, items, url):
    prefix, _, name = url.rpartition("/")
    store = open_store(prefix or ".")
    writer = await rustac.GeoparquetWriter.open(items, name, store=store)
    writer.add_collection(collection)
    await writer.finish()
    log.info("wrote %s (%d items)", url, len(items))
