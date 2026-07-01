# cdh-data-pipeline

Shared plumbing for turning source datasets into analysis-ready cloud-optimized
outputs for the climate data hub. The general hope is that many datasets can be
ingested into the hub storage using this by submitting a recipe. It is heavily
inspired by the pangeo forge pipeline.

## Layout

- `src/cdh_data_pipeline/` — the library (the boring, identical part)
  - `storage.py` — obstore store factory + source raster reading
  - `zarr.py` — zarr writing (compression codec + `write_zarr`)
  - `cog.py` — COG conversion (`make_cog`, `COG_OPTS`)
- `recipes/` — one script per ingested dataset (the part that differs)
  - `glw4.py` — GLW4 livestock density
  - `mapspam.py` — MapSPAM 2020 V2r2 crop statistics
  - `examples/` — runnable reference recipes that aren't ingested (write
    locally, demo a technique); copy one as a starting point. e.g.
    `berkeley_tavg.py` (multiscale store with per-level chunking for point +
    animated-map reads).

A recipe imports the helpers, declares its own source mapping + dataset
assembly, and calls `write_zarr` / `make_cog`. Adding a dataset = a new file in
`recipes/` (or `recipes/examples/` for a demo that isn't ingested).

## Running

Run a recipe from the **repo root** — `run(...)` executes its build steps (any
`fetch` → zarr → COGs) in order and writes a zarr store + COGs to the recipe's
`OUTPUT`.

```sh
# For example
uv run recipes/glw4.py
uv run --env-file .env recipes/mapspam.py   # needs $DATAVERSE_TOKEN (see Credentials)
```

Re-running overwrites the outputs; a `fetch` step (if any) skips source files
already downloaded.

## Adding a dataset

Copy `recipes/glw4.py` (the minimal example) and edit four things:

1. **Config** — `INPUT` (source path/URL), `OUTPUT` (local or `s3://`/`gs://`),
   and the source naming (a `SRC` template or a `src()` helper).
2. **Assembly** — read sources with `open_raster`, build an `xarray.Dataset`,
   set `title`/`source` attrs.
3. **`build_zarr()` / `build_cogs()`** — call `write_zarr(ds, url, encoding)`
   (GeoZarr tagging + vlen string coords are handled for you) and
   `make_cog(srcs, names, units)` per COG (pass 1-element lists for
   single-band).
4. **Entry point** — `run(build_zarr, build_cogs)`, prepending a `fetch` step if
   the source must be downloaded first.

No registration step — a recipe is just a runnable script that calls the shared
helpers.

## Credentials

Credentials come from the **environment**

- **`OUTPUT` (obstore)** reads credentials from **environment variables only** —
  it does _not_ parse `~/.aws/credentials` / `AWS_PROFILE`. For S3:
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`,
  `AWS_REGION`); for GCS: `GOOGLE_APPLICATION_CREDENTIALS` (service-account JSON
  path). To use an AWS profile, export it into the env first, e.g.
  `aws configure export-credentials --profile NAME --format env` (eval'd / piped
  to `source` in fish).
- **`INPUT` (GDAL/rasterio)** is a separate layer with its own vars (the same
  `AWS_*` / `GOOGLE_APPLICATION_CREDENTIALS`). Public `https://` sources need
  none; registration-gated sources need their own token (e.g. mapspam's
  `DATAVERSE_TOKEN` for Harvard Dataverse).

Recommended: export the vars in your shell. To keep project-local vars in a file
instead, `uv` loads one natively (no extra dependency):

```sh
uv run --env-file .env recipes/glw4.py    # or once: export UV_ENV_FILE=.env
```

See `.env.example` for the vars recipes use — copy it to `.env` and fill in.
`.env` is gitignored; never commit real keys.

## Dev

```sh
uv sync                 # create the env
uv run ruff check .     # lint (+ import sort)
uv run ruff format .    # format
uv run ty check         # type check
prek run --all-files    # all hooks
```
