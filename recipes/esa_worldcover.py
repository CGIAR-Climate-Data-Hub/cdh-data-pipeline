"""ESA WorldCover 2021 v200 land cover -> VRT mosaic over the AWS Open Data COGs.

Nothing is copied (the map is ~117 GB). The bucket has no STAC, but it ships
the 3x3 degree tile grid and tile names follow it, so STAC items are built from
the grid and the VRT is laid out without opening a tile. ``map.vrt`` is the
11-class land cover map (uint8, 0 = nodata, colour table and class codes from
the tiles). ``class-codes.json`` maps class code to name and colour.

Run from the repo root: uv run recipes/esa_worldcover.py
"""

import json
import urllib.request

import rasterio

from cdh_data_pipeline import run, write_json, write_vrt
from cdh_data_pipeline.mosaic import _vsi

INPUT = "https://esa-worldcover.s3.amazonaws.com"
OUTPUT = "s3://digital-atlas/cdh/data/esa-worldcover-2021-v200"
TILE = 36000  # pixels per 3 degree tile side; 10 m


def item(ll_tile):
    """STAC item for a tile named by its lower-left corner, such as ``N00E006``."""
    lat = int(ll_tile[1:3]) * (1 if ll_tile[0] == "N" else -1)
    lon = int(ll_tile[4:7]) * (1 if ll_tile[3] == "E" else -1)
    bbox = [lon, lat, lon + 3, lat + 3]
    ring = [[lon, lat], [lon + 3, lat], [lon + 3, lat + 3], [lon, lat + 3], [lon, lat]]
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
        ],
        "id": ll_tile,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "bbox": bbox,
        "properties": {
            "datetime": "2021-01-01T00:00:00Z",
            "proj:epsg": 4326,
            "proj:shape": [TILE, TILE],
            "proj:transform": [3 / TILE, 0, lon, 0, -3 / TILE, lat + 3],
        },
        "links": [],
        "assets": {
            "map": {
                "href": f"{INPUT}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ll_tile}_Map.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"],
            }
        },
    }


def build():
    with urllib.request.urlopen(f"{INPUT}/esa_worldcover_grid.geojson") as r:
        grid = json.load(r)
    items = [item(f["properties"]["ll_tile"]) for f in grid["features"]]
    write_vrt(f"{OUTPUT}/map.vrt", {"map": items}, asset="map")
    with rasterio.open(_vsi(items[0]["assets"]["map"]["href"])) as src:
        colors = src.colormap(1)
        legend = (line.split(maxsplit=1) for line in src.tags()["legend"].splitlines())
        codes = {
            code: {
                "name": name,
                "color": "#{:02x}{:02x}{:02x}".format(*colors[int(code)][:3]),
            }
            for code, name in legend
        }
    write_json(f"{OUTPUT}/class-codes.json", codes)


if __name__ == "__main__":
    run(build)
