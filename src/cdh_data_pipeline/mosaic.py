"""Virtual mosaics over tiled rasters: VRT (band per tile set) and GDAL tile index.

Tiles are described either by STAC items or by raster paths/URLs. Items are laid
out from their ``proj:`` metadata without opening a file; paths are opened once.
Nothing is copied, the outputs reference the tiles in place.
"""

import json
import math
import tempfile
from pathlib import Path

import geopandas as gpd
import rasterio
import rio_vrt
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy  # ty: ignore[unresolved-import]
from shapely.geometry import box, shape

from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import put_file

# GDAL virtual filesystem prefix per URL scheme; GTI/VRT need paths GDAL can open.
_VSI = {"s3": "/vsis3/", "gs": "/vsigs/", "az": "/vsiaz/", "abfs": "/vsiaz/"}
# numpy dtype names -> GDAL names, for the GTI layer metadata.
_GDAL_TYPE = {
    "int8": "Int8",
    "uint8": "Byte",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "float32": "Float32",
    "float64": "Float64",
}


def write_vrt(url, bands, *, asset="data"):
    """Write a VRT at ``url`` with one band per entry of ``bands``.

    ``bands`` maps band name to that band's source: one raster path/URL (stacked
    as is, e.g. one file per year), a list of raster paths/URLs (mosaicked, each
    opened once by rio-vrt), or a list of STAC items (mosaicked by GDAL's STACIT
    driver from ``proj:`` metadata, no file opened). Mosaics are written as
    ``<stem>-<band>.vrt`` siblings and the bands stacked on top; a single tile set
    writes just that mosaic at ``url``.

    Sources must share pixel size, data type and nodata; mismatches raise rather
    than being silently misplaced, recast or unmasked. Any GDAL 2+ reader opens the
    result and sees the tiles' overview levels as virtual overviews. HTTPS tile
    locations disable sidecar probing (~15 requests per tile on servers without
    directory listings).
    """
    prefix, _, name = url.rpartition("/")
    sources = {
        b: [src] if isinstance(src, (str, Path)) else list(src)
        for b, src in bands.items()
    }
    _check_publishable(url, [s for srcs in sources.values() for s in srcs], asset)
    lone = all(isinstance(src, (str, Path)) for src in bands.values())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp, name)
        if lone:  # one raster per band: stack them as they are, no sibling files
            _build_vrt(
                out,
                [_vsi(str(srcs[0])) for srcs in sources.values()],
                mosaic=False,
                names=tuple(sources),
            )
        else:
            parts = []
            for band, srcs in sources.items():
                part = out if len(sources) == 1 else Path(tmp, f"{out.stem}-{band}.vrt")
                if not srcs:
                    raise ValueError(f"{part.stem}: no sources")
                if isinstance(srcs[0], dict):
                    _stacit_to_vrt(srcs, asset, part)
                else:
                    _build_vrt(part, [_vsi(str(s)) for s in srcs], mosaic=True)
                parts.append(part)
            if len(parts) > 1:
                _build_vrt(
                    out, parts, mosaic=False, relative=True, names=tuple(sources)
                )
        for part in Path(tmp).iterdir():
            put_file(f"{prefix or '.'}/{part.name}", part)
    log.info("wrote %s (%d bands)", url, len(bands))


def _stacit_to_vrt(items, asset, out):
    """Mosaic one asset across STAC items via STACIT and serialise it as a VRT."""
    features = [
        {
            **i,
            "assets": {
                asset: i["assets"][asset] | {"href": _vsi(i["assets"][asset]["href"])}
            },
        }
        for i in items
    ]
    body = json.dumps({"type": "FeatureCollection", "features": features}).encode()
    with MemoryFile(body, ext=".json") as mem:
        with rasterio.open(f'STACIT:"{mem.name}"', ASSET=asset, MAX_ITEMS="0") as src:
            rio_copy(src, out, driver="VRT")
    _finish_vrt(out, _probe(features[0]["assets"][asset]["href"]))


def _build_vrt(out, sources, *, mosaic, relative=False, names=None):
    """rio-vrt mosaic or stack; it assumes square pixels and drops nodata, so fix both."""
    tiles = [_probe(src) for src in sources]
    if len(sources) == 1:  # rio-vrt crashes on a single input; wrap it directly
        with rasterio.open(sources[0]) as raster:
            rio_copy(raster, out, driver="VRT")
    else:
        for src, tile in zip(sources[1:], tiles[1:]):
            for key in ("res", "dtype", "nodata"):  # the VRT carries one of each
                if not _same(key, tile[key], tiles[0][key]):
                    raise ValueError(
                        f"{src}: {key} {tile[key]} differs from {sources[0]}: {tiles[0][key]}"
                    )
        rio_vrt.build_vrt(
            out,
            [str(s) for s in sources],
            mosaic=mosaic,
            relative=relative,
            res=tiles[0]["res"],
        )
    _finish_vrt(out, tiles[0], names)


def _same(key, a, b):
    """Resolution tolerates float noise; nodata is a sentinel and must match exactly."""
    if key == "res":
        return all(math.isclose(x, y, rel_tol=1e-9) for x, y in zip(a, b))
    if key == "nodata" and isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _probe(source):
    """Open one tile (or VRT) once for what the mosaic must inherit from it."""
    with rasterio.open(source) as src:
        return {
            "res": src.res,
            "nodata": src.nodata,
            "overviews": src.overviews(1),
            "dtype": _GDAL_TYPE[src.dtypes[0]],
            "bands": src.count,
            "crs": src.crs,
        }


def _finish_vrt(out, tile, names=None):
    """Set what the builders drop, and advertise the tiles' overview levels virtually.

    Virtual overviews store nothing: GDAL reads the sources at reduced resolution,
    which in turn hit the COGs' internal overviews. They make the levels visible
    to readers that ask (QGIS, tile servers, ``vrt://...?ovr=N``).
    """
    resampling = (
        Resampling.average if tile["dtype"].startswith("Float") else Resampling.nearest
    )
    with rasterio.Env(VRT_VIRTUAL_OVERVIEWS="YES"), rasterio.open(out, "r+") as vrt:
        if tile["nodata"] is not None:
            vrt.nodata = tile["nodata"]
        if names:
            vrt.descriptions = names
        if tile["overviews"]:
            vrt.build_overviews(tile["overviews"], resampling)


def write_gti(url, tiles, *, asset="data"):
    """Write a GDAL tile index (GeoPackage) at ``url`` over ``tiles``.

    ``tiles`` are STAC items (footprints from metadata) or raster paths/URLs (each
    opened once for its bounds). Every GDAL 3.8+ opens the result as a mosaic via
    ``GTI:<url>``, rasterio wheels included. The layer carries the tiles' CRS,
    raster metadata and overview levels so GDAL neither warps nor probes a tile on
    open. Prefer ``write_vrt`` unless the tile count needs GTI's spatial index or
    the footprint layer itself is wanted.
    """
    _check_publishable(url, tiles, asset)
    df = (
        _footprints_from_items(tiles, asset)
        if isinstance(tiles[0], dict)
        else _footprints_from_files(tiles)
    )
    tile = _probe(df.location.iloc[0])
    df = df.to_crs(tile["crs"])
    meta = {
        "SRS": tile["crs"].to_string(),
        "RESX": str(tile["res"][0]),
        "RESY": str(tile["res"][1]),
        "BAND_COUNT": str(tile["bands"]),
        "DATA_TYPE": tile["dtype"],
    }
    if tile["nodata"] is not None:
        meta["NODATA"] = str(tile["nodata"])
    # virtual overview levels, read through the tiles' own overviews
    for i, factor in enumerate(tile["overviews"]):
        meta[f"OVERVIEW_{i}_FACTOR"] = str(factor)
    # GPKG is sqlite: write locally, then put
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp, url.rpartition("/")[2])
        df.to_file(local, driver="GPKG", layer_metadata=meta)
        put_file(url, local)
    log.info("wrote %s (%d tiles)", url, len(df))


def _footprints_from_items(items, asset):
    return gpd.GeoDataFrame(
        {
            "id": [i["id"] for i in items],
            "location": [_vsi(i["assets"][asset]["href"]) for i in items],
        },
        geometry=[shape(i["geometry"]) for i in items],
        crs="OGC:CRS84",
    )


def _footprints_from_files(paths):
    paths = [_vsi(str(p)) for p in paths]
    footprints, crs = [], None
    for p in paths:
        with rasterio.open(p) as src:
            if crs is None:
                crs = src.crs
            elif src.crs != crs:  # GTI needs one CRS
                raise ValueError(f"{p} is in {src.crs}, index is in {crs}")
            footprints.append(box(*src.bounds))
    ids = [Path(p).stem for p in paths]
    return gpd.GeoDataFrame(
        {"id": ids, "location": paths}, geometry=footprints, crs=crs
    )


def _check_publishable(url, sources, asset):
    """A mosaic in object storage must not point at files that exist only here."""
    if "://" not in url:
        return
    for source in sources:
        href = (
            source["assets"][asset]["href"] if isinstance(source, dict) else str(source)
        )
        if "://" not in href and not href.startswith("/vsi"):
            raise ValueError(f"{url} would reference local files, e.g. {href}")


def _vsi(href):
    """Location string GDAL can open from anywhere: absolute local path or /vsi path."""
    scheme, sep, rest = href.partition("://")
    if not sep:
        return href if href.startswith("/vsi") else str(Path(href).resolve())
    if scheme in ("http", "https"):
        return f"/vsicurl?empty_dir=yes&url={href}"
    return _VSI[scheme] + rest if scheme in _VSI else href  # GDAL strings pass through
