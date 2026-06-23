# cdh-data-pipeline
Shared plumbing for turning source datasets into analysis-ready cloud-optimized
outputs for the climate data hub. The general hope is that many datasets can be
ingested into the hub storage using this by submitting a recipe. It is heavily
inspired by the pangeo forge pipeline.

A recipe imports the helpers, declares its own source mapping + dataset
assembly, and calls `write_zarr` / `make_cog`. Adding a dataset = a new file in
`recipes/`.

## Dev

```sh
uv sync                 # create the env
uv run ruff check .     # lint (+ import sort)
uv run ruff format .    # format
uv run ty check         # type check
prek run --all-files    # all hooks
```
