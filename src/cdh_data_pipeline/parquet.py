"""Parquet / GeoParquet writer."""

from pathlib import Path

import geopandas as gpd

from cdh_data_pipeline.storage import open_fs

# write_statistics is already True by default, so it is not repeated here.
_PARQUET_OPTS = dict(
    compression="zstd",
    compression_level=9,
    row_group_size=100_000,
    write_page_index=True,
)


_HILBERT = "__hilbert__"


def write_parquet(df, url, *, sort=True, **kwargs):
    """Write a DataFrame to ``url`` as Parquet, or a GeoDataFrame as GeoParquet 1.1.

    Geometry goes out as WKB with a bbox covering column, Hilbert-sorted so each row
    group's bbox stays tight enough for readers to skip on a spatial filter. Page
    indexes are written so readers can also skip pages within a row group.

    ``sort`` sets the physical row order that row-group statistics are built from.
    ``True`` (default) Hilbert-sorts a GeoDataFrame and leaves a plain DataFrame
    alone; ``False`` writes rows as given. A list of column names sorts by those
    columns so readers skip on an attribute filter, and for a GeoDataFrame applies
    Hilbert order within each group — which only buys back spatial skipping when a
    key value holds more rows than ``row_group_size``, otherwise row groups straddle
    key values and every bbox covers the full extent.

    Extra kwargs reach ``pyarrow.parquet.write_table`` and override the defaults,
    e.g. ``partition_cols`` for a hive layout. Hive is tabular-only: geopandas
    writes a single file and cannot partition.
    """
    fs = open_fs(url)
    if fs is None:  # obstore mkdirs for zarr stores; pyarrow will not
        Path(url).parent.mkdir(parents=True, exist_ok=True)
    opts = {**_PARQUET_OPTS, **kwargs}
    keys = [] if sort is True else list(sort or [])
    if isinstance(df, gpd.GeoDataFrame):
        if "partition_cols" in opts:
            raise ValueError(
                "partition_cols is tabular-only; geopandas cannot write a hive "
                "dataset. Partition by hand into one GeoParquet per prefix."
            )
        if sort is not False:
            df = (
                df.assign(**{_HILBERT: df.geometry.hilbert_distance()})
                .sort_values([*keys, _HILBERT])
                .drop(columns=_HILBERT)
            )
        # required by write_covering_bbox; caller still wins
        opts = {"schema_version": "1.1.0", "write_covering_bbox": True, **opts}
    elif keys:
        df = df.sort_values(keys)
    df.to_parquet(url, index=False, filesystem=fs, **opts)
    print(f"wrote {url} ({len(df)} rows)")
