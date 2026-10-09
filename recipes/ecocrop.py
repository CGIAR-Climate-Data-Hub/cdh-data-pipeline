"""FAO EcoCrop crop characteristic database -> one Parquet table.

Source is the CSV export in OpenCLIM/ecocrop (no DOI of its own; their Zenodo
DOI covers the model code). Pinned to the last commit that touched the file.

Cleaning only, no curation: UTF-8, lowercase FAO column codes, typed numbers,
comma-joined categoricals as list<string>. No rows dropped or values corrected,
so the known bad rows (latitudes > 90, TOPMN of 160) are passed through.

Run from the repo root: uv run recipes/ecocrop.py
"""

from pathlib import Path

import pandas as pd

from cdh_data_pipeline import download, run, write_parquet

INPUT = Path("input/ecocrop/EcoCrop_DB.csv")
OUTPUT = "s3://digital-atlas/cdh/data/ecocrop/ecocrop.parquet"

COMMIT = "ea43ecd418fd1a1ea08d32c7dce64b7fae02dec4"  # 2022-03-01
URL = f"https://raw.githubusercontent.com/OpenCLIM/ecocrop/{COMMIT}/EcoCrop_DB.csv"

# Comma-joined multi-value columns; no vocabulary item contains a comma.
# Listed by hand: scientificname and auth also contain commas that are not
# separators ("Pinus merkusii, island provenances").
LISTS = [
    "syno",
    "comname",
    "lifo",
    "habi",
    "lispa",
    "phys",
    "cat",
    "plat",
    "text",
    "textr",
    "dra",
    "drar",
    "photo",
    "cliz",
    "abitol",
    "abisus",
    "intri",
    "prosy",
]


def fetch():
    download(URL, INPUT)


def build_parquet():
    # pandas infers the numeric columns and treats both "NA" and "" as null.
    df = pd.read_csv(INPUT, encoding="cp1252")
    df.columns = df.columns.str.lower()
    for c in LISTS:
        df[c] = df[c].str.split(r"\s*,\s*", regex=True)
    write_parquet(df.convert_dtypes(), OUTPUT, sort="ecoportcode")


if __name__ == "__main__":
    run(fetch, build_parquet)
