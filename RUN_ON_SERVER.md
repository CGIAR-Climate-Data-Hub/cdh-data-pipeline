# Running the global SPEI build on a server

`recipes/spei.py` builds the monthly SPEI cube on the CHIRPS v3 grid
(0.05 deg, 60 N - 60 S, 2400 x 7200). The inputs are read over HTTPS, so the
server needs internet access but no copy of the input cubes.

## 1. Install

```bash
git clone https://github.com/CGIAR-Climate-Data-Hub/cdh-data-pipeline.git
cd cdh-data-pipeline
git checkout spei
uv sync                    # add --extra hf only on the machine that publishes
uv run pytest tests/test_spei.py -q
```

## 2. Settings

Put them in `.env` (see `.env.example`) or export them. Only `SPEI_OUTPUT` is
needed for the default build.

| Variable | Default | Meaning |
|---|---|---|
| `SPEI_OUTPUT` | `output/<dataset-id>` | Folder for the store and its sidecars. Use a local disk with ~60 GB free, not a synced folder (OneDrive, Dropbox). |
| `SPEI_PET` | `hargreaves` | `hargreaves` (CHIRTS-ERA5 Tmax/Tmin) or `agera5` (AgERA5 Penman-Monteith ET0). The dataset id, and so the store name, contains this value. |
| `SPEI_WINDOWS` | `3` | Windows in months, comma-separated, e.g. `3,6,12`. |
| `SPEI_THROUGH` | latest complete month | `YYYY-MM`: last month to build. |
| `SPEI_WORKERS` | `8` | Parallel chunk reads per input. |
| `SPEI_BBOX` | whole grid | `W,S,E,N`, only for test runs. Leave unset for the global build. |
| `CDSAPI_KEY` | `~/.cdsapirc` | CDS key; needed only with `SPEI_PET=agera5`. |

Example `.env`:

```bash
SPEI_OUTPUT=/data/cdh/spei-chirps-v3-hargreaves-monthly
SPEI_PET=hargreaves
SPEI_WINDOWS=3
```

## 3. Run

Use `tmux` or `screen` so the build survives a lost SSH session:

```bash
tmux new -s spei
uv run --env-file .env recipes/spei.py 2>&1 | tee -a spei.log
```

A bare run does, in order: `inspect`, `create_store`, `monthly_inputs`, `spei`,
`verify`, `write_sources`, `write_card`, `write_cdh`. It never runs `update`,
`publish_plan` or `publish`.

### Stopping and resuming

Stop at any time (Ctrl-C, reboot, lost connection). To resume, run **the same
command again**:

* `inputs.json` pins the input revisions (Hugging Face commit SHAs) and the
  months. Once a tile is written, `inspect` keeps these pins, so the resumed
  build reads exactly the same data even if CHIRPS was updated in between.
* `tiles.json` lists every finished tile for each step. Finished tiles are
  skipped; the tile that was in progress is redone from the start.

To run one step only: `uv run --env-file .env recipes/spei.py monthly_inputs`.

To start again from nothing, delete `SPEI_OUTPUT` (store, `inputs.json`,
`tiles.json`).

### What to expect

* The grid has 192 tiles of 300 x 300 cells. Each tile with land streams its
  daily inputs (CHIRPS and both CHIRTS variables) for 1981 to now and needs
  about 1-2 GB of memory. In the pilots (laptop, home connection) one land
  tile took 2-3.5 minutes with `hargreaves` and 1-2 minutes with `agera5`,
  almost all of it reading the inputs; fitting and writing take seconds. With
  well over 100 land tiles, plan for several hours to most of a day.
* Tiles of open ocean are recorded with `land_cells: 0` and skipped quickly.
* Each tile is logged as it is finished, with its land cells and the days with
  Tmin > Tmax. Those counts per cell are written to `diagnostics/`.
* `verify` writes `verify.json` and sets `build_complete` on the store only if
  every check passes: all tiles done, values within [-3, 3], no SPEI outside
  the CHIRPS land mask, no month without any value.

## 4. Later

* **More windows:** set `SPEI_WINDOWS=3,6,12` and run
  `recipes/spei.py create_store spei verify write_sources write_card write_cdh`.
  The monthly inputs already in the store are reused.
* **Monthly update:** `recipes/spei.py update verify write_sources write_card`
  re-pins the inputs, adds new months and recomputes the last 6 months with the
  stored distribution parameters (no refit).
* **Refit** (for example after a CHIRPS revision inside 1991-2020): remove the
  `monthly_inputs` and `spei_NN` sections from `tiles.json`, then do a bare run.
* **Publish** only after review: `publish_plan` shows what would be sent;
  `publish` sends it (needs `uv sync --extra hf` and a Hugging Face token).
