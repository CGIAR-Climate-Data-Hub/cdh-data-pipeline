"""Parquet / GeoParquet writer."""

import geopandas as gpd
import numpy as np
import pandas as pd

from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import open_fs

_PARQUET_OPTS = {
    "compression": "zstd",
    "row_group_size": 100_000,
    "write_page_index": True,
}


def write_parquet(df, url, *, sort=True, **kwargs):
    """Write a DataFrame as Parquet, or a GeoDataFrame as GeoParquet 1.1.

    GeoDataFrames get a bbox column and are Hilbert-sorted so readers can skip
    row groups on a spatial filter.

    ``sort``: ``True`` Hilbert-sorts geometry, ``False`` keeps row order, and a
    list of columns sorts by those first (then Hilbert, for geometry).
    Extra kwargs go to pyarrow, e.g. ``partition_cols`` (plain DataFrames only).
    """
    fs = open_fs(url)
    opts = {"index": False, **_PARQUET_OPTS, **kwargs}
    if opts.get("compression") == "zstd":
        opts.setdefault("compression_level", 9)
    if isinstance(sort, str):
        sort = [sort]
    keys = [] if sort is True else list(sort or [])
    if isinstance(df, gpd.GeoDataFrame):
        if "partition_cols" in opts:
            raise ValueError(
                "partition_cols is not supported for GeoDataFrames. "
                "Partition by hand into one GeoParquet per prefix, if needed."
            )
        if sort is not False:
            geometry = df.geometry.name

            def sort_key(column):
                if column.name != geometry:
                    return column
                g = gpd.GeoSeries(column, crs=df.crs)
                ok = ~(g.is_empty | g.isna())
                # hilbert_distance fails on empty geometry; sort those rows last.
                out = pd.Series(np.nan, index=g.index)
                if ok.any():
                    out.loc[ok] = g[ok].hilbert_distance()
                return out

            df = df.sort_values([*keys, geometry], key=sort_key)
        # write_covering_bbox needs schema 1.1.0
        opts = {"schema_version": "1.1.0", "write_covering_bbox": True, **opts}
    elif keys:
        df = df.sort_values(keys)
    log.info("writing %s (%d rows)", url, len(df))
    df.to_parquet(url, filesystem=fs, **opts)
    log.info("wrote %s", url)
