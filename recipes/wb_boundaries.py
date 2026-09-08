"""World Bank Official Boundaries -> one GeoParquet per admin layer.

Currently World Bank Global Administrative Divisions (GAD) v2;
Data Catalog dataset 0038272 version 3, released 2026-07-14.

Run from the repo root: uv run recipes/wb_boundaries.py
"""

import urllib.parse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from cdh_data_pipeline import download, run, write_parquet

INPUT = Path("input/wb_boundaries")
OUTPUT = "s3://digital-atlas/cdh/data/wb-boundaries-gad"

BASE = "https://datacatalogfiles.worldbank.org/ddh-published/0038272/"
GPKG = (
    "3/DR0095370/World Bank Official Boundaries (GeoPackage)/"
    "World Bank Official Boundaries - {}.gpkg"
)

# output name : (source name, rows per row group).
LAYERS = {
    # adm0 is countries + disputed areas; filter on wb_status
    # adm1 & 2 do not include disputed areas
    "adm0": ("Admin 0_all_layers", 50),
    "adm1": ("Admin 1", 250),
    "adm2": ("Admin 2", 2000),
    "ocean-mask": ("Ocean Mask", 50),
}
# Code crosswalk (HASC, GAUL, P-codes) and alternate names, joinable to adm1/adm2
# on adm1cd_c / adm2cd_c.
ATTRS = {
    "adm1-attributes": "DR0095373/WB_GAD_adm1_additional_columns.csv",
    "adm2-attributes": "DR0095374/WB_GAD_adm2_additional_columns.csv",
}


def fetch():
    """Download sources into INPUT (skips any already present)."""
    files = {f"{n}.gpkg": GPKG.format(s) for n, (s, _) in LAYERS.items()}
    files |= {f"{n}.csv": p for n, p in ATTRS.items()}
    for name, path in files.items():
        download(BASE + urllib.parse.quote(path), INPUT / name)


def fix_japan(df):
    """Upstream bug: Japan's wb_status is "Other" and its sov_iso_a3 "Member State"."""
    if "sov_iso_a3" in df:
        df.loc[df.sov_iso_a3 == "Member State", "sov_iso_a3"] = "JPN"
    if "wb_status" in df:
        df.loc[df.iso_a3 == "JPN", "wb_status"] = "Member State"
    return df


def load_layer(name):
    df = gpd.read_file(INPUT / f"{name}.gpkg")
    # Column names and string values carry a UTF-8 BOM prefix.
    df.columns = df.columns.str.removeprefix("﻿").str.lower()
    df = df.replace(r"^﻿", "", regex=True)
    # make_valid leaves valid rows untouched.
    bad = ~df.is_valid
    df.loc[bad, "geometry"] = df.geometry[bad].make_valid()
    return fix_japan(df)


def load_attrs(name):
    df = pd.read_csv(INPUT / f"{name}.csv", dtype=str)
    # drop ArcGIS export artefacts
    df = df.drop(columns=["Layer", "Shape_Leng", "Shape_Area"], errors="ignore")
    df.columns = df.columns.str.lower()
    # cast GAUL codes to integers
    gaul = df.filter(like="gaul_").columns
    df[gaul] = df[gaul].astype("Int64").replace(0, pd.NA)
    return fix_japan(df)


def build_parquet():
    for name, (_, rows) in LAYERS.items():
        write_parquet(load_layer(name), f"{OUTPUT}/{name}.parquet", row_group_size=rows)
    for name in ATTRS:
        df = load_attrs(name)
        # Sorted by the join key so a code lookup skips row groups.
        write_parquet(df, f"{OUTPUT}/{name}.parquet", sort=df.columns[0])


if __name__ == "__main__":
    run(fetch, build_parquet)
