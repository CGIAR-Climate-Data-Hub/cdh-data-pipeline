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
- `recipes/` — one script per dataset (the part that differs)
  - `glw4.py` — GLW4 livestock density
  - `mapspam.py` — MapSPAM 2020 V2r2 crop statistics

A recipe imports the helpers, declares its own source mapping + dataset
assembly, and calls `write_zarr` / `make_cog`. Adding a dataset = a new file in
`recipes/`.

## Credentials

Credentials come from the **environment**

- **`OUTPUT` (obstore)** reads credentials from **environment variables only** —
  it does *not* parse `~/.aws/credentials` / `AWS_PROFILE`. For S3:
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
