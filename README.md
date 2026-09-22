# cdh-data-pipeline

Shared code for turning source datasets into analysis-ready cloud-optimized
(ARCO) outputs for the climate data hub. Each dataset is added as a recipe that
can be run to ingest it into hub storage. The pipeline is heavily inspired by
Pangeo Forge.

## Layout

- `src/cdh_data_pipeline/`: the shared library
  - `recipe.py`: `run` (timed build steps, exits 1 on failure; pass step names
    to run a subset, such as `uv run recipes/mapspam.py build_cogs`) and the
    shared `log` used by every writer and available to recipes
  - `download.py`: cached source downloads (`download`, `download_dataverse`)
  - `storage.py`: obstore store and filesystem factories, plus source raster
    reading
  - `zarr.py`: zarr writing (compression codec and `write_zarr`)
  - `parquet.py`: Parquet and GeoParquet 1.1 writing (`write_parquet`)
  - `stac.py`: static STAC collections (`read_stac_collection`) and STAC
    GeoParquet snapshots (`write_stac_geoparquet`)
  - `cog.py`: COG conversion (`make_cog`, `write_cog`). Pass `cog_options=` to
    override GDAL creation options for a call.
- `recipes/`: one script per ingested dataset
  - `glw4.py`: GLW4 livestock density
  - `mapspam.py`: MapSPAM 2020 V2r2 crop statistics
  - `wb_boundaries.py`: World Bank Official Boundaries (admin 0 to 2 and ocean
    mask) as GeoParquet, plus the admin 1 and 2 attribute tables. Admin 0
    includes the disputed NDLSA areas. Filter on `wb_status`.
  - `examples/`: runnable reference recipes that write locally and demonstrate a
    technique. Copy one as a starting point, such as `berkeley_tavg.py`
    (multiscale store with per-level chunking for point + animated-map reads).

A recipe imports the helpers, declares its source mapping and dataset assembly,
then calls `run(...)` to run the build steps. To add a dataset, create a new
file in `recipes/`. Use `recipes/examples/` for a demo that is not ingested.

## Running

Run a recipe from the repo root. `run(...)` executes its build steps in order,
such as `fetch`, zarr, and COGs, and writes the outputs to the recipe's
`OUTPUT`.

```sh
# For example
uv run recipes/glw4.py
uv run --env-file .env recipes/mapspam.py   # needs $DATAVERSE_TOKEN (see Credentials)
```

Re-running overwrites the outputs. A `fetch` step skips source files that have
already been downloaded locally.

## Adding a dataset

Copy `recipes/glw4.py`, the minimal example, and edit four things:

1. **Config**: `INPUT` (source path or URL), `OUTPUT` (local or
   `s3://`/`gs://`), and the source naming (`SRC` template or `src()` helper).
2. **Assembly**: read sources with `open_raster`, build an `xarray.Dataset`, and
   set the `title` and `source` attributes.
3. **`build_zarr()` / `build_cogs()`**: call `write_zarr(ds, url, encoding)`.
   GeoZarr tagging and variable-length string coordinates are handled for you.
   Call `write_cog(url, srcs, names, units)` for each COG. Pass one-element
   lists for single-band COGs.
4. **Entry point**: call `run(build_zarr, build_cogs)`. Add a `fetch` step first
   if the source needs to be downloaded.

There is no registration step. A recipe is just a runnable script that calls the
shared helpers.

Zarr outputs must use the conventional `.zarr` suffix so overwrite cleanup stays
scoped to a store prefix.

## Credentials

Credentials come from the environment.

- **`OUTPUT` (obstore)** reads credentials from environment variables only. It
  does not parse `~/.aws/credentials` or `AWS_PROFILE`. For S3, use:
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`,
  `AWS_REGION`); for GCS: `GOOGLE_APPLICATION_CREDENTIALS` (service-account JSON
  path). To use an AWS profile, export it into the env first, e.g.
  `aws configure export-credentials --profile NAME --format env` (eval'd / piped
  to `source` in fish).
- **`INPUT` (GDAL/rasterio)** is a separate layer with its own variables. It
  uses the same `AWS_*` and `GOOGLE_APPLICATION_CREDENTIALS` variables. Public
  `https://` sources need no credentials. Registration-gated sources need their
  own token, such as mapspam's `DATAVERSE_TOKEN` for Harvard Dataverse.

Recommended: export the vars in your shell. To keep project-local vars in a file
instead, `uv` loads one natively (no extra dependency):

```sh
uv run --env-file .env recipes/glw4.py    # or once: export UV_ENV_FILE=.env
```

See `.env.example` for the variables recipes use. Copy it to `.env` and fill it
in. `.env` is gitignored; never commit real keys.

## Dev

```sh
uv sync                 # create the env
uv run ruff check .     # lint (+ import sort)
uv run ruff format .    # format
uv run ty check         # type check
prek run --all-files    # all hooks
```
