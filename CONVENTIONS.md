# Data Lake Conventions

Conventions for everything published under `s3://digital-atlas/cdh/`.
Proposals below are the working standard; open questions are marked **discuss**.

## Naming

Two rules cover everything:

1. **Object keys (files, directories, dataset ids): lowercase kebab-case.**
   `mapspam-2020-v2r2`, `crop-codes.json`, `glw4-2020.zarr`.
   URL-safe without encoding, no case-sensitivity foot-guns on S3, `-` reads
   as a word boundary to humans and tooling alike.
2. **In-data identifiers (zarr variables, dims, coords, attrs, JSON keys):
   snake_case.** `physical_area`, `crop_name`, `long_name`. CF convention;
   works unquoted in Python (`ds.physical_area`) and R (`df$physical_area`).

These compose: in a filename, `-` separates fields and `_` only ever appears
inside an identifier field, so `spam2020-physical_area-all.tif` splits
unambiguously on `-` into dataset / variable / technology.

Never camelCase. Never spaces. Never uppercase in object keys.

Machine identifiers vs labels: keep the source's stable codes as the
machine-facing values (zarr coords, COG band names — e.g. SPAM crop codes
`whea`, `pige`) and carry human-readable names alongside as labels
(`crop_name` coord, `long_name` band tag, `*-codes.json`). Both products of a
dataset must use the same code vocabulary so one catalog domain field serves
all assets.

## Dataset layout

One prefix per dataset version, fully self-contained:

```
cdh/data/<dataset-id>/
  <dataset-id>.zarr        # analysis cube (chunked/sharded for expected reads)
  cog/                     # per-layer COGs for display and GIS users
    <dataset>-<field>[-<field>].tif
  *.json                   # dataset-level metadata (e.g. crop-codes.json)
```

- Everything a consumer needs lives under the one prefix. Deleting the prefix
  deletes the dataset; copying it moves the dataset intact. No orphans, no
  cross-prefix references.
- **COGs live in `cog/` inside the dataset prefix, not in a separate
  top-level tree.** Zarr and COGs are two renditions of the same data; the
  catalog entry points at both as assets of one item. A parallel `cogs/`
  hierarchy would split each dataset across two prefixes and rot
  independently.
- **Destinations are strings, not `Path`s.** Recipes build output urls by
  interpolating a string prefix (`f"{OUTPUT}/cog/..."`). `pathlib` flattens
  `s3://bucket` to `s3:/bucket`, which reads as a relative local path, so a
  `Path` url would write to the wrong place. `Path` is for genuinely local
  files only, such as a download cache.
- **discuss:** should the zarr store name repeat the dataset id exactly?
  Today `glw4-2020/glw4-2020.zarr` does but
  `mapspam-2020-v2r2/spam2020-v2r2.zarr` doesn't. Proposal: yes, store name =
  `<dataset-id>.zarr` — self-describing if it ever leaves the bucket.

## Versioning

Version is part of the dataset id, flat, siblings:

```
cdh/data/mapspam-2020-v2r2/
cdh/data/mapspam-2020-v2r3/
```

- Use the **upstream producer's version string** (`v2r2` is IFPRI's, not
  ours). Don't invent our own version axis.
- Published URLs are immutable: a new upstream version is a new prefix, never
  an overwrite of the old one.
- Pipeline bugfixes that reprocess the *same* upstream version overwrite in
  place via a recipe rerun (recipes clear the store first). The recipe in
  this repo is the source of truth; outputs are reproducible, so in-place
  regeneration is safe and doesn't warrant a version bump.
- **No `latest/` alias or duplicated objects.** "Latest" is a catalog
  concern: express it as a link/field in the catalog entry, not by copying
  data. Aliases drift and cached URLs lie.
- **discuss:** if a dataset accrues many versions, do we ever nest
  (`mapspam-2020/v2r2/`)? Proposal: no — flat ids keep URLs one level deep
  and the catalog handles grouping; revisit only if listing the data root
  becomes unwieldy.

## Metadata & catalog

- CF-style metadata (units, long_name, title, institution, references) lives
  *inside* the zarr attrs and COG band tags — the data is self-describing.
- Dataset-level sidecars (code↔name tables, provenance) are JSON files at the
  dataset root, written by the recipe (`write_json`).
- STAC is static and colocated: item/collection JSON in the dataset prefix
  with relative asset hrefs, a root `catalog.json` at `cdh/data/`. A search
  API, if ever needed, harvests these; the static files stay the source of
  truth.

## General practices

- **Recipes are the only writers.** Every object in the lake is traceable to
  a recipe in this repo; nothing is hand-uploaded or hand-edited.
- Rerunning a recipe is the only mutation path, and it rewrites a store
  wholesale rather than patching objects.
- Chunk/shard for the expected read pattern and say what that pattern is in a
  comment next to the encoding (see `recipes/mapspam.py`).
- Prefer one store per dataset over many small stores; prefer variables and
  dimensions over encoding facets into filenames — filenames are for the
  things object storage forces apart (COG singles, sidecars).

## Current deviations

- `mapspam-2020-v2r2/spam2020-v2r2.zarr` and its COG names use `spam2020`
  while the prefix uses `mapspam-2020` — pick one at first publish (nothing
  is in the bucket yet for this dataset).
