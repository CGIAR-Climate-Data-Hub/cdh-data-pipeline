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

NUMERIC = [
    "ecoportcode", "topmn", "topmx", "tmin", "tmax", "ropmn", "ropmx", "rmin",
    "rmax", "phopmn", "phopmx", "phmin", "phmax", "latopmn", "latopmx", "latmn",
    "latmx", "altmx", "ktmpr", "ktmp", "gmin", "gmax",
]  # fmt: skip
# Comma-joined multi-value columns. No vocabulary item contains a comma.
LISTS = [
    "syno", "comname", "lifo", "habi", "lispa", "phys", "cat", "plat", "text",
    "textr", "dra", "drar", "photo", "cliz", "abitol", "abisus", "intri", "prosy",
]  # fmt: skip


def fetch():
    download(URL, INPUT)


def build_parquet():
    df = pd.read_csv(INPUT, encoding="cp1252", dtype=str, na_values=["NA", ""])
    df.columns = df.columns.str.lower()
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric)
    for c in LISTS:
        df[c] = (
            df[c]
            .str.split(",")
            .map(lambda x: [s.strip() for s in x], na_action="ignore")
        )
    write_parquet(df.convert_dtypes(), OUTPUT, sort="ecoportcode")


if __name__ == "__main__":
    run(fetch, build_parquet)
