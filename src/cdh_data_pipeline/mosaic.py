"""VRT and GTI (GDAL tile index) mosaics that reference tiles in place.

Tiles are STAC items (laid out from ``proj:`` metadata) or raster paths/URLs.
"""

import json
import math
import tempfile
from pathlib import Path

import geopandas as gpd
import rasterio
import rio_vrt
from rasterio.dtypes import dtype_rev, typename_fwd
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy  # ty: ignore[unresolved-import]
from shapely.geometry import box, shape

from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import put_file

# URL scheme -> GDAL virtual filesystem prefix
_VSI = {"s3": "/vsis3/", "gs": "/vsigs/", "az": "/vsiaz/", "abfs": "/vsiaz/"}


def write_vrt(url, bands, *, asset="data", compute_stats=False):
    """Write a VRT at ``url`` with one band per entry of ``bands``.

    ``bands`` maps band name to a source: one raster path/URL, a list of them,
    or a list of STAC items. With several bands, lists are mosaicked into
    ``<stem>-<band>.vrt`` siblings that the main VRT stacks.

    Band stats come from the STAC items' band statistics when every item has
    them. ``compute_stats`` fills the rest from overviews, which reads every tile.
    """
    _check_publishable(url, bands.values(), asset)
    prefix, _, name = url.rpartition("/")
    names = tuple(bands)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp, name)
        if len(bands) == 1:
            first, band_stats = _mosaic(bands[names[0]], asset, out)
            stats = [band_stats]
        elif all(isinstance(src, (str, Path)) for src in bands.values()):
            files = [_vsi(str(src)) for src in bands.values()]
            _build(out, files, mosaic=False)
            first, stats = files[0], [None] * len(files)
        else:  # rio-vrt's relative flag is all-or-nothing, so every band gets a sibling
            parts = []
            for band, src in bands.items():
                part = Path(tmp, f"{out.stem}-{band}.vrt")
                part_first, part_stats = _mosaic(src, asset, part)
                _add_metadata(part, part_first, (band,), [part_stats], compute_stats)
                parts.append(part)
            _build(out, parts, mosaic=False, relative=True)
            first, stats = parts[0], [_read_stats(p) for p in parts]
        _add_metadata(out, first, names, stats, compute_stats)
        for part in Path(tmp).iterdir():
            put_file(f"{prefix or '.'}/{part.name}", part)
    log.info("wrote %s (%d bands)", url, len(bands))


def _mosaic(src, asset, out):
    """Write one band's source (a raster, list of rasters or STAC items) to ``out``.

    Returns the first raster's path, which the VRT copies nodata and overviews
    from, and the band stats from STAC (None for plain rasters).
    """
    if isinstance(src, (str, Path)):
        src = [src]
    if not src:
        raise ValueError(f"{out.stem}: no sources")
    if isinstance(src[0], dict):
        _mosaic_stac(src, asset, out)
        return _vsi(_href(src[0], asset)), _stac_stats(src, asset)
    files = [_vsi(str(s)) for s in src]
    _build(out, files, mosaic=True)
    return files[0], None


def _mosaic_stac(items, asset, out):
    """Mosaic ``asset`` across STAC items with GDAL's STACIT driver, saved as VRT."""
    features = [
        i | {"assets": {asset: i["assets"][asset] | {"href": _vsi(_href(i, asset))}}}
        for i in items
    ]
    body = json.dumps({"type": "FeatureCollection", "features": features}).encode()
    with (
        MemoryFile(body, ext=".json") as mem,
        rasterio.open(f'STACIT:"{mem.name}"', ASSET=asset, MAX_ITEMS="0") as src,
    ):
        rio_copy(src, out, driver="VRT")


def _stac_stats(items, asset):
    """Combine each tile's band stats into mosaic stats; None if any tile lacks them.

    Tiles are weighted by valid pixels. Without ``valid_percent`` they are
    weighted by size, so mean and stddev are marked approximate.
    """
    tiles = []
    for i in items:
        a, props = i["assets"][asset], i["properties"]
        # STAC 1.1 ``bands``, or the 1.0 raster extension's ``raster:bands``
        bands = next(
            (d[k] for d in (a, props) for k in ("bands", "raster:bands") if d.get(k)),
            None,
        )
        stats = bands[0].get("statistics", {}) if bands else {}
        if not {"minimum", "maximum", "mean", "stddev"} <= stats.keys():
            log.warning("%s: no band stats for %s, skipping", i["id"], asset)
            return None
        h, w = a.get("proj:shape") or props["proj:shape"]
        tiles.append((h * w * stats.get("valid_percent", 100) / 100, stats))
    n = sum(count for count, _ in tiles)
    mean = sum(count * s["mean"] for count, s in tiles) / n
    var = sum(
        count * (s["stddev"] ** 2 + (s["mean"] - mean) ** 2) for count, s in tiles
    )
    out = {
        "STATISTICS_MINIMUM": min(s["minimum"] for _, s in tiles),
        "STATISTICS_MAXIMUM": max(s["maximum"] for _, s in tiles),
        "STATISTICS_MEAN": mean,
        "STATISTICS_STDDEV": math.sqrt(var / n),
    }
    if any("valid_percent" not in s for _, s in tiles):
        out["STATISTICS_APPROXIMATE"] = "YES"
    return out


def _read_stats(path):
    """Band 1 ``STATISTICS_*`` tags of a raster."""
    with rasterio.open(path) as src:
        return {k: v for k, v in src.tags(1).items() if k.startswith("STATISTICS_")}


def _build(out, files, *, mosaic, relative=False):
    """Mosaic or stack ``files`` with rio-vrt.

    All must share resolution, dtype and nodata: rio-vrt takes them from the first
    file, so a mismatch would silently corrupt values.
    """
    if len(files) == 1:  # rio-vrt crashes on a single input
        with rasterio.open(files[0]) as src:
            rio_copy(src, out, driver="VRT")
        return
    props = []
    for f in files:
        with rasterio.open(f) as src:
            props.append((src.res, src.dtypes[0], str(src.nodata)))  # str: NaN == NaN
    for f, (res, dtype, nodata) in zip(files, props):
        if (
            not all(map(math.isclose, res, props[0][0]))
            or (dtype, nodata) != props[0][1:]
        ):
            raise ValueError(
                f"{f}: {res, dtype, nodata} differs from {files[0]}: {props[0]}"
            )
    # rio-vrt assumes square pixels unless given res
    rio_vrt.build_vrt(
        out, [str(f) for f in files], mosaic=mosaic, relative=relative, res=props[0][0]
    )


def _add_metadata(vrt_path, first_source, names=None, stats=None, compute_stats=False):
    """Set nodata (from ``first_source``), band names, stats and virtual overviews.

    rio-vrt drops nodata when stacking. Virtual overviews store nothing; reads fall
    through to the COGs' overviews. ``stats`` holds one dict or None per band.
    """
    first = _raster_info(first_source)
    resampling = (
        Resampling.average if first["dtype"].startswith("Float") else Resampling.nearest
    )
    with (
        rasterio.Env(VRT_VIRTUAL_OVERVIEWS="YES"),
        rasterio.open(vrt_path, "r+") as vrt,
    ):
        if first["nodata"] is not None:
            vrt.nodata = first["nodata"]
        if names:
            vrt.descriptions = names
        if first["overviews"]:
            vrt.build_overviews(first["overviews"], resampling)
        stats = list(stats or [])
        missing = [band for band, s in enumerate(stats, 1) if not s]
        if compute_stats and missing:
            # update_stats(approx=True) doesn't persist to a VRT, so write tags
            for band, s in zip(missing, vrt.stats(indexes=missing, approx=True)):
                stats[band - 1] = {
                    "STATISTICS_MINIMUM": s.min,
                    "STATISTICS_MAXIMUM": s.max,
                    "STATISTICS_MEAN": s.mean,
                    "STATISTICS_STDDEV": s.std,
                    "STATISTICS_APPROXIMATE": "YES",
                }
        for band, band_stats in enumerate(stats, 1):
            if band_stats:
                vrt.update_tags(band, **band_stats)


def _raster_info(path):
    """Read the raster properties a mosaic inherits from its first tile."""
    with rasterio.open(path) as src:
        return {
            "res": src.res,
            "nodata": src.nodata,
            "overviews": src.overviews(1),
            "dtype": typename_fwd[dtype_rev[src.dtypes[0]]],
            "bands": src.count,
            "crs": src.crs,
        }


def write_gti(url, tiles, *, asset="data"):
    """Write a GTI GeoPackage at ``url`` over STAC items or raster paths/URLs.

    Open with ``GTI:<url>`` (GDAL 3.8+). Prefer ``write_vrt`` unless you have
    many tiles or need the footprint layer.
    """
    _check_publishable(url, tiles, asset)
    df = (
        _footprints_from_items(tiles, asset)
        if isinstance(tiles[0], dict)
        else _footprints_from_files(tiles)
    )
    tile = _raster_info(df.location.iloc[0])
    df = df.to_crs(tile["crs"])
    # stops GDAL warping or opening tiles on open
    meta = {
        "SRS": tile["crs"].to_string(),
        "RESX": str(tile["res"][0]),
        "RESY": str(tile["res"][1]),
        "BAND_COUNT": str(tile["bands"]),
        "DATA_TYPE": tile["dtype"],
    }
    if tile["nodata"] is not None:
        meta["NODATA"] = str(tile["nodata"])
    for i, factor in enumerate(tile["overviews"]):
        meta[f"OVERVIEW_{i}_FACTOR"] = str(factor)
    # GPKG is SQLite, so write locally then upload
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp, url.rpartition("/")[2])
        df.to_file(local, driver="GPKG", layer_metadata=meta)
        put_file(url, local)
    log.info("wrote %s (%d tiles)", url, len(df))


def _footprints_from_items(items, asset):
    return gpd.GeoDataFrame(
        {
            "id": [i["id"] for i in items],
            "location": [_vsi(_href(i, asset)) for i in items],
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
    """Raise if a remote mosaic would reference local files.

    ``sources`` holds rasters, STAC items, or lists of either.
    """
    if "://" not in url:
        return
    for src in sources:
        for s in src if isinstance(src, (list, tuple)) else [src]:
            href = _href(s, asset)
            if "://" not in href and not href.startswith("/vsi"):
                raise ValueError(f"{url} would reference local files, e.g. {href}")


def _href(source, asset):
    """Path/URL of a raster, or of ``asset`` in a STAC item."""
    return source["assets"][asset]["href"] if isinstance(source, dict) else str(source)


def _vsi(href):
    """Convert a URL or path to a GDAL path (/vsi prefix or absolute path)."""
    scheme, sep, rest = href.partition("://")
    if not sep:
        return href if href.startswith("/vsi") else str(Path(href).resolve())
    if scheme in ("http", "https"):
        # empty_dir=yes skips sidecar probing, ~15 requests per tile
        return f"/vsicurl?empty_dir=yes&url={href}"
    return _VSI[scheme] + rest if scheme in _VSI else href
