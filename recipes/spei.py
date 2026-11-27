"""Monthly SPEI on the CHIRPS v3 0.05 deg grid (60 N - 60 S) -> Zarr cube.

Run from the repo root: uv run --env-file .env recipes/spei.py
A bare run does the whole build and can be stopped and started again: every
tile is recorded in ``tiles.json`` when it is finished. Individual steps:

    uv run recipes/spei.py inspect          # pin the input revisions, find the months
    uv run recipes/spei.py monthly_inputs   # daily inputs -> monthly precip and PET
    uv run recipes/spei.py spei             # fit and transform, per window
    uv run recipes/spei.py verify
    uv run recipes/spei.py update           # new and recent months (never in a bare run)
    uv run recipes/spei.py publish_plan     # what publish would send (never in a bare run)
    uv run recipes/spei.py publish

Method: see :mod:`cdh_data_pipeline.spei` (ported from MOSAIC 27_spei.py).
SPEI = standard normal quantile of a generalized logistic distribution fitted per
cell and calendar month to the window sums of P - PET over 1991-2020.

Settings come from environment variables:

``SPEI_PET``      ``hargreaves`` (default; CHIRTS-ERA5 Tmax/Tmin) or ``agera5``
                  (AgERA5 FAO-56 Penman-Monteith ET0). The PET source is part
                  of the dataset id, so the two never share a store.
``SPEI_OUTPUT``   folder for the store and its sidecars.
``SPEI_BBOX``     ``W,S,E,N`` to build only part of the grid (pilots).
``SPEI_WINDOWS``  comma-separated windows in months, default ``3``.
``SPEI_THROUGH``  ``YYYY-MM``: last month to build (default: the latest month
                  complete in every input).
``SPEI_WORKERS``  parallel reads per input (default 8).
``CDSAPI_KEY``    CDS key for AgERA5 (or ``~/.cdsapirc``).

Adding a window later (e.g. ``SPEI_WINDOWS=3,6,12``) adds its arrays and runs
only the ``spei`` step for it; the monthly inputs are reused from the store.
"""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cdh_data_pipeline import GridSpec, hf, log, run, write_json
from cdh_data_pipeline import spei as S
from cdh_data_pipeline import spei_cube as C

PET = os.environ.get("SPEI_PET", "hargreaves")
if PET not in S.PET_METHODS:
    raise SystemExit(f"SPEI_PET must be one of {S.PET_METHODS}, not {PET!r}")

DATASET = f"spei-chirps-v3-{PET}-monthly"
OUTPUT = os.environ.get("SPEI_OUTPUT", f"output/{DATASET}")
STORE_NAME = f"{DATASET}.zarr"
URL = f"{OUTPUT}/{STORE_NAME}"
INPUTS = f"{OUTPUT}/inputs.json"
TILE_LOG = f"{OUTPUT}/tiles.json"

BBOX = (
    tuple(float(v) for v in os.environ["SPEI_BBOX"].split(","))
    if os.environ.get("SPEI_BBOX")
    else None
)
WINDOWS = tuple(int(w) for w in os.environ.get("SPEI_WINDOWS", "3").split(","))
THROUGH = os.environ.get("SPEI_THROUGH")
WORKERS = int(os.environ.get("SPEI_WORKERS", "8"))

#: WMO climatological standard normal; same as MOSAIC.
CALIBRATION = (1991, 2020)
#: Months recomputed by ``update`` behind the last one, because CHIRPS revises
#: its latest months after first release.
RECOMPUTE_MONTHS = 6

CHIRPS_REPO = "aaguilar90/chirps-v3-daily-rnl"
CHIRPS_STORE = "chirps-v3-daily-rnl.zarr"
CHIRTS_REPO = "aaguilar90/chirts-era5-daily"
CHIRTS_STORE = "chirts-era5-daily.zarr"

#: The target grid is the CHIRPS grid.
CHIRPS_GRID = GridSpec(height=2400, width=7200, res=0.05, north=60.0, nodata=-9999.0)

HF_REPO = os.environ.get("SPEI_HF_REPO", f"aaguilar90/{DATASET}")
HF_SIDECARS = ("README.md", "sources.json", "inputs.json", f"{DATASET}.yaml")

PET_TEXT = {
    "hargreaves": "Hargreaves-Samani reference evapotranspiration (FAO-56 eq. 52) "
    "from CHIRTS-ERA5 daily Tmax and Tmin",
    "agera5": "AgERA5 FAO-56 Penman-Monteith reference evapotranspiration, "
    "regridded bilinearly from 0.1 deg",
}
DISTRIBUTION = (
    "generalized logistic (the three-parameter log-logistic of the SPEI papers)"
)
FITTING = "unbiased probability-weighted moments (L-moments), Hosking 1990"


# ------------------------------------------------------------- helpers


def _area():
    """Target grid of this run and its place in the global CHIRPS grid."""
    if BBOX is None:
        return CHIRPS_GRID, C.Window(0, CHIRPS_GRID.height, 0, CHIRPS_GRID.width)
    return C.subgrid(CHIRPS_GRID, BBOX)


def _inputs() -> dict:
    if not Path(INPUTS).exists():
        raise SystemExit(f"{INPUTS} missing: run the inspect step first")
    return json.loads(Path(INPUTS).read_text())


def _months(pins: dict) -> np.ndarray:
    first, last = pins["months"]
    return np.arange(
        np.datetime64(first, "M"), np.datetime64(last, "M") + np.timedelta64(1, "M")
    )


def _open(pins: dict) -> dict:
    """Open the pinned inputs."""
    out = {}
    g = C.open_http_group(pins["chirps"]["url"])
    out["precip"] = C.DailyCube(g, "precip", CHIRPS_GRID)
    if PET == "hargreaves":
        g = C.open_http_group(pins["chirts"]["url"])
        out["tmax"] = C.DailyCube(g, "Tmax", CHIRPS_GRID)
        out["tmin"] = C.DailyCube(g, "Tmin", CHIRPS_GRID)
    else:
        out["et0"] = C.Agera5ET0(
            pins["agera5"]["url"], {"Authorization": f"Bearer {C.cds_key()}"}
        )
    return out


def _resolve_inputs() -> dict:
    """Pin each input to its current revision and find the common months."""
    pins = {"pet_method": PET, "resolved": _today()}
    rev = C.hf_revision(CHIRPS_REPO)
    url = C.hf_store_url(CHIRPS_REPO, rev, CHIRPS_STORE)
    p = C.DailyCube(C.open_http_group(url), "precip", CHIRPS_GRID)
    pins["chirps"] = {
        "repo": CHIRPS_REPO,
        "revision": rev,
        "url": url,
        "time_coverage": [str(p.dates[0]), str(p.dates[-1])],
    }
    sets = [S.complete_months(p.dates)]
    if PET == "hargreaves":
        rev = C.hf_revision(CHIRTS_REPO)
        url = C.hf_store_url(CHIRTS_REPO, rev, CHIRTS_STORE)
        t = C.DailyCube(C.open_http_group(url), "Tmax", CHIRPS_GRID)
        pins["chirts"] = {
            "repo": CHIRTS_REPO,
            "revision": rev,
            "url": url,
            "time_coverage": [str(t.dates[0]), str(t.dates[-1])],
        }
        sets.append(S.complete_months(t.dates))
    else:
        e = C.Agera5ET0(C.AGERA5_ARCO, {"Authorization": f"Bearer {C.cds_key()}"})
        pins["agera5"] = {
            "url": C.AGERA5_ARCO,
            "variable": C.AGERA5_ET0,
            "time_coverage": [str(e.dates[0]), str(e.dates[-1])],
            "history": e.attrs.get("history"),
            "references": e.attrs.get("references"),
        }
        sets.append(S.complete_months(e.dates))
    months = S.common_months(*sets)
    if THROUGH:
        months = months[months <= np.datetime64(THROUGH, "M")]
    pins["months"] = [str(months[0]), str(months[-1])]
    return pins


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _root_attrs(grid: GridSpec, months) -> dict:
    return {
        "Conventions": "CF-1.10",
        "id": DATASET,
        "title": f"SPEI monthly on the CHIRPS v3 0.05 deg grid ({PET} PET)",
        "institution": "Alliance of Bioversity International and CIAT",
        "source": f"CHIRPS v3.0 daily precipitation (RNL) minus {PET_TEXT[PET]}",
        "references": "Vicente-Serrano, Begueria & Lopez-Moreno (2010), J. Climate "
        "23:1696-1718; Begueria et al. (2014), Int. J. Climatol. 34:3001-3023",
        "license": "CC-BY-4.0",
        "spatial_resolution_deg": grid.res,
        "geospatial_lat_min": grid.south,
        "geospatial_lat_max": grid.north,
        "geospatial_lon_min": grid.west,
        "geospatial_lon_max": grid.east,
        "time_coverage_start": str(months[0]),
        "time_coverage_end": str(months[-1]),
        "spei_windows": list(WINDOWS),
        "calibration_period": f"{CALIBRATION[0]}-{CALIBRATION[1]}",
        "distribution": DISTRIBUTION,
        "fitting_method": FITTING,
        "pet_method": PET_TEXT[PET],
        "build_complete": False,
    }


def _var_attrs() -> dict:
    out = {
        "precip": {
            "long_name": "monthly precipitation, CHIRPS v3.0 (RNL)",
            "standard_name": "precipitation_amount",
            "units": "mm",
            "cell_methods": "time: sum",
            "comment": "Missing for a cell if any day of the month is missing.",
        },
        "pet": {
            "long_name": f"monthly reference evapotranspiration: {PET_TEXT[PET]}",
            "standard_name": "water_potential_evaporation_amount",
            "units": "mm",
            "cell_methods": "time: sum",
            "pet_method": PET_TEXT[PET],
        },
    }
    for w in WINDOWS:
        out[C.spei_name(w)] = {
            "long_name": f"Standardized Precipitation Evapotranspiration Index, {w}-month",
            "cell_methods": f"time: sum (interval: {w} months) of precip - pet",
            "calibration_period": f"{CALIBRATION[0]}-{CALIBRATION[1]}",
            "distribution": DISTRIBUTION,
            "fitting_method": FITTING,
            "pet_method": PET_TEXT[PET],
            "comment": "Labelled by the last month of the window (bounds in "
            f"time_bnds_{w:02d}). Clipped to [-3, 3]. Missing where any month of "
            "the window is missing or the distribution could not be fitted.",
        }
    return out


def _rows(tile: C.Window, area: C.Window) -> tuple[slice, slice]:
    """Slices of ``tile`` inside the store, whose origin is ``area``."""
    return (
        slice(tile.row0 - area.row0, tile.row1 - area.row0),
        slice(tile.col0 - area.col0, tile.col1 - area.col0),
    )


def _monthly(src: dict, months, tile: C.Window, lat) -> tuple:
    """Monthly precip and PET for one tile, plus diagnostics."""
    precip = C.monthly_from_daily(src["precip"], months, tile, WORKERS)
    diag = {}
    if not np.isfinite(precip).any():
        return precip, None, diag
    if PET == "hargreaves":
        pet, diag = C.monthly_hargreaves(
            src["tmax"], src["tmin"], months, tile, lat, WORKERS
        )
    else:
        glat, glon = CHIRPS_GRID.coords()
        pet = src["et0"].monthly_on(
            months, glat[tile.row0 : tile.row1], glon[tile.col0 : tile.col1]
        )
    return precip, pet, diag


def _save_diag(key: str, diag: dict) -> dict:
    """Write the Tmin > Tmax counts of a tile; return a small summary."""
    if not diag:
        return {}
    d = Path(OUTPUT) / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"tmin_gt_tmax_{key}.npz",
        days=diag["tmin_gt_tmax_days"],
        max_excess=diag["tmin_gt_tmax_max_excess"],
        years=np.array(sorted(diag["by_year"])),
        by_year=np.array([diag["by_year"][y] for y in sorted(diag["by_year"])]),
    )
    return {"tmin_gt_tmax_days": int(diag["tmin_gt_tmax_days"].sum())}


def _tile_spei(root, tile, area, months, w, rows=None):
    """SPEI for one tile and window, fitted on the full record.

    With ``rows`` (a slice of months), only those months are transformed, with
    the stored parameters: that is how ``update`` adds months without refitting.
    """
    ry, rx = _rows(tile, area)
    name = C.spei_name(w)
    if rows is None:
        d = root["precip"][:, ry, rx].astype("f8") - root["pet"][:, ry, rx]
        tot = S.rolling_sum(d, w)
        # Transform with the parameters exactly as stored (float32), so a later
        # update, which reads them back, reproduces these values bit for bit.
        params = S.fit_params(tot, months, CALIBRATION).astype("f4")
        values = S.spei_from_params(tot, months, params.astype("f8"))
        root[name][:, ry, rx] = C.encode_spei(values)
        for i, p in enumerate(("xi", "alpha", "k")):
            root[f"{name}_{p}"][:, ry, rx] = params[:, i]
        return int(np.isfinite(params[:, 0]).any(0).sum())
    start = max(rows.start - (w - 1), 0)
    d = root["precip"][start : rows.stop, ry, rx].astype("f8")
    d -= root["pet"][start : rows.stop, ry, rx]
    tot = S.rolling_sum(d, w)[rows.start - start :]
    params = np.stack(
        [root[f"{name}_{p}"][:, ry, rx] for p in ("xi", "alpha", "k")], axis=1
    ).astype("f8")
    values = S.spei_from_params(tot, months[rows], params)
    root[name][rows, ry, rx] = C.encode_spei(values)
    return None


# --------------------------------------------------------------- steps


def inspect():
    """Pin the input revisions and the months to build, in ``inputs.json``.

    Once tiles have been written the pins are kept, so a resumed build reads
    exactly the same inputs; ``update`` is the only step that moves them.
    """
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    if Path(INPUTS).exists() and any(C.TileLog(TILE_LOG).data.values()):
        pins = _inputs()
        log.info(
            "keeping the pinned inputs (tiles already written): %s", pins["months"]
        )
    else:
        pins = _resolve_inputs()
        write_json(INPUTS, pins)
    grid, area = _area()
    src = _open(pins)  # checks that every input is on the CHIRPS cell centres
    log.info(
        "grid %dx%d (%s..%s N, %s..%s E), %d tile(s); months %s..%s; PET %s; inputs %s",
        grid.height,
        grid.width,
        grid.south,
        grid.north,
        grid.west,
        grid.east,
        len(C.tiles(area)),
        *pins["months"],
        PET,
        {
            k: (v.get("revision") or v.get("history", ""))[:12]
            for k, v in pins.items()
            if isinstance(v, dict)
        },
    )
    del src


def create_store():
    """Create the empty cube (or add the arrays of a new window)."""
    pins = _inputs()
    months = _months(pins)
    grid, _ = _area()
    root = C.create_store(
        C.cube_store_url(URL),
        grid,
        months,
        WINDOWS,
        _root_attrs(grid, months),
        _var_attrs(),
    )
    if root["time"].shape[0] != len(months):
        raise SystemExit(
            f"store has {root['time'].shape[0]} months, inputs.json {len(months)}: "
            "run update instead"
        )


def monthly_inputs():
    """Monthly precip and PET, tile by tile; finished tiles are skipped."""
    pins = _inputs()
    months = _months(pins)
    _, area = _area()
    root = C.open_store_rw(URL)
    tl = C.TileLog(TILE_LOG)
    done = tl.done("monthly_inputs")
    src = _open(pins)
    glat, _ = CHIRPS_GRID.coords()
    todo = [t for t in C.tiles(area) if t.key not in done]
    log.info("monthly_inputs: %d tile(s) to do, %d done", len(todo), len(done))
    for n, tile in enumerate(todo, 1):
        precip, pet, diag = _monthly(src, months, tile, glat[tile.row0 : tile.row1])
        ry, rx = _rows(tile, area)
        land = int(np.isfinite(precip).any(0).sum())
        info = {"land_cells": land}
        if land:
            root["precip"][:, ry, rx] = precip.astype("f4")
            root["pet"][:, ry, rx] = pet.astype("f4")
            both = np.isfinite(precip) & np.isfinite(pet)
            info["cells_with_both"] = int(both.any(0).sum())
            info.update(_save_diag(tile.key, diag))
        tl.mark("monthly_inputs", tile.key, info)
        log.info("  tile %s (%d/%d): %s", tile.key, n, len(todo), info)


def spei():
    """Fit and transform each window, tile by tile; finished tiles are skipped."""
    pins = _inputs()
    months = _months(pins)
    _, area = _area()
    root = C.open_store_rw(URL)
    tl = C.TileLog(TILE_LOG)
    ready = tl.done("monthly_inputs")
    for w in WINDOWS:
        step = C.spei_name(w)
        done = tl.done(step)
        todo = [t for t in C.tiles(area) if t.key in ready and t.key not in done]
        log.info("%s: %d tile(s) to do, %d done", step, len(todo), len(done))
        for tile in todo:
            fitted = 0
            if ready[tile.key]["land_cells"]:
                fitted = _tile_spei(root, tile, area, months, w)
            tl.mark(step, tile.key, {"fitted_cells": fitted})
    C.consolidate(URL)


def verify():
    """Check range, land mask and empty months; set ``build_complete`` if all pass."""
    pins = _inputs()
    months = _months(pins)
    _, area = _area()
    root = C.open_store_rw(URL)
    tl = C.TileLog(TILE_LOG)
    report, problems = {}, []
    all_tiles = C.tiles(area)
    for step in ["monthly_inputs"] + [C.spei_name(w) for w in WINDOWS]:
        missing = [t.key for t in all_tiles if t.key not in tl.done(step)]
        if missing:
            problems.append(f"{step}: {len(missing)} tile(s) not done")
    for w in WINDOWS:
        name = C.spei_name(w)
        per_month = np.zeros(len(months), np.int64)
        stats = {
            k: 0
            for k in (
                "land",
                "spei",
                "land_without_spei",
                "spei_without_land",
                "out_of_range",
                "negative_precip",
                "negative_pet",
            )
        }
        for tile in all_tiles:
            ry, rx = _rows(tile, area)
            precip = root["precip"][:, ry, rx]
            land = np.isfinite(precip).any(0)
            if not land.any():
                continue
            raw = root[name][:, ry, rx]
            ok = raw != C.SPEI_FILL
            has = ok.any(0)
            stats["land"] += int(land.sum())
            stats["spei"] += int(has.sum())
            stats["land_without_spei"] += int((land & ~has).sum())
            stats["spei_without_land"] += int((has & ~land).sum())
            stats["out_of_range"] += int((ok & (np.abs(raw) > 300)).sum())
            stats["negative_precip"] += int((precip < 0).sum())
            stats["negative_pet"] += int((root["pet"][:, ry, rx] < 0).sum())
            per_month += ok.reshape(len(months), -1).sum(1)
        empty = [str(m) for m, c in zip(months[w - 1 :], per_month[w - 1 :]) if c == 0]
        stats["empty_months"] = empty
        report[name] = stats
        for k in (
            "spei_without_land",
            "out_of_range",
            "negative_precip",
            "negative_pet",
        ):
            if stats[k]:
                problems.append(f"{name}: {k} = {stats[k]}")
        if empty:
            problems.append(
                f"{name}: {len(empty)} month(s) without any value: {empty[:5]}"
            )
    report["problems"] = problems
    write_json(f"{OUTPUT}/verify.json", report)
    log.info("verify: %s", json.dumps(report))
    if problems:
        raise RuntimeError("verify failed: " + "; ".join(problems))
    root.attrs["build_complete"] = True
    C.consolidate(URL)


def write_sources():
    """``sources.json``: input revisions and spans, method, code version."""
    pins = _inputs()
    write_json(
        f"{OUTPUT}/sources.json",
        {
            "dataset": DATASET,
            "inputs": {k: v for k, v in pins.items() if isinstance(v, dict)},
            "pet_method": PET_TEXT[PET],
            "months": pins["months"],
            "windows": list(WINDOWS),
            "calibration_period": list(CALIBRATION),
            "distribution": DISTRIBUTION,
            "fitting_method": FITTING,
            "bbox": list(BBOX) if BBOX else None,
            "code": {
                "repo": "https://github.com/CGIAR-Climate-Data-Hub/cdh-data-pipeline",
                "commit": _git_commit(),
            },
            "tiles": "tiles.json",
        },
    )


def write_card():
    """Hugging Face dataset card (README.md)."""
    pins = _inputs()
    grid, _ = _area()
    windows = ", ".join(f"`{C.spei_name(w)}` ({w}-month)" for w in WINDOWS)
    text = (
        f"""---
license: cc-by-4.0
pretty_name: SPEI monthly, CHIRPS v3 grid ({PET} PET)
tags: [drought, spei, climate, precipitation, evapotranspiration, zarr]
---

# SPEI monthly on the CHIRPS v3 0.05 degree grid ({PET} PET)

The Standardized Precipitation Evapotranspiration Index (SPEI) for every month
from {pins["months"][0]} to {pins["months"][1]}, on the CHIRPS v3 grid
(0.05 degree, EPSG:4326, {grid.south} to {grid.north} latitude).

* Variables: {windows}; monthly `precip` and `pet` (mm); and the fitted
  distribution parameters `<spei>_xi`, `<spei>_alpha`, `<spei>_k` per calendar month.
* Water balance: CHIRPS v3.0 daily precipitation (RNL) minus {PET_TEXT[PET]},
  summed to months, then over the window.
* Distribution: {DISTRIBUTION}, fitted per cell and calendar month to the
  {CALIBRATION[0]}-{CALIBRATION[1]} window sums by {FITTING}.
* SPEI is the standard normal quantile of the fitted probability, clipped to
  [-3, 3]. Each value is labelled by the last month of its window; the window
  bounds are in `time_bnds_NN`.
* Storage: Zarr v3; SPEI as int16 with `scale_factor` 0.01 and `_FillValue`
  -32768; precip and pet as float32. Zstandard level 5. Chunks
  ({C.TIME_CHUNK} months, 300, 300).

A month is missing for a cell when any of its days is missing in an input.
The latest 2-3 months of CHIRPS are provisional and are recomputed by later
updates.

## Inputs

* CHIRPS v3.0 daily RNL: https://huggingface.co/datasets/{CHIRPS_REPO}
  (revision `{pins["chirps"]["revision"][:12]}`)
"""
        + (
            f"""* CHIRTS-ERA5 daily Tmax/Tmin: https://huggingface.co/datasets/{CHIRTS_REPO}
  (revision `{pins["chirts"]["revision"][:12]}`)
"""
            if PET == "hargreaves"
            else """* AgERA5 (Copernicus Climate Change Service), FAO-56 Penman-Monteith ET0,
  doi:10.24381/cds.6c68c9bb
"""
        )
        + """
## Citation

Vicente-Serrano, S. M., Begueria, S. & Lopez-Moreno, J. I. (2010). A multiscalar
drought index sensitive to global warming: the Standardized Precipitation
Evapotranspiration Index. J. Climate 23, 1696-1718.
Begueria, S. et al. (2014). Standardized precipitation evapotranspiration index
(SPEI) revisited. Int. J. Climatol. 34, 3001-3023.
Funk, C. et al. (2015). The climate hazards infrared precipitation with
stations. Sci. Data 2, 150066.
"""
    )
    Path(OUTPUT, "README.md").write_text(text, encoding="utf-8")
    log.info("wrote %s/README.md", OUTPUT)


def write_cdh():
    """CDH v0.4.1 metadata record (``<dataset>.yaml``), in the style of the other cubes."""
    import yaml

    pins = _inputs()
    grid, _ = _area()
    today = _today()
    derived = [
        {
            "url": f"https://huggingface.co/datasets/{CHIRPS_REPO}",
            "title": "CHIRPS v3.0 daily RNL Zarr",
        }
    ]
    if PET == "hargreaves":
        derived.append(
            {
                "url": f"https://huggingface.co/datasets/{CHIRTS_REPO}",
                "title": "CHIRTS-ERA5 daily Zarr",
            }
        )
    else:
        derived.append({"url": C.AGERA5_ARCO, "title": "AgERA5 ARCO Zarr (ECMWF)"})
    variables = [
        {
            "name": C.spei_name(w),
            "description": f"{w}-month SPEI, clipped to [-3, 3]. int16, scale_factor 0.01.",
            "data_type": "int16",
            "unit": "1",
            "nodata": C.SPEI_FILL,
        }
        for w in WINDOWS
    ] + [
        {
            "name": "precip",
            "description": "Monthly CHIRPS v3.0 precipitation.",
            "data_type": "float32",
            "unit": "mm",
        },
        {
            "name": "pet",
            "description": f"Monthly {PET_TEXT[PET]}.",
            "data_type": "float32",
            "unit": "mm",
        },
    ]
    rec = {
        "cdh_schema_version": "v0.4.1",
        "extensions": [
            (
                "https://cgiar-climate-data-hub.github.io/cdh-metadata-standard/v0.4.1/"
                "extensions/cdh/schema.json"
            )
        ],
        "id": DATASET,
        "title": f"SPEI monthly on the CHIRPS v3 0.05 degree grid ({PET} PET)",
        "description": (
            "Standardized Precipitation Evapotranspiration Index from CHIRPS v3.0 "
            f"precipitation minus {PET_TEXT[PET]}. {DISTRIBUTION.capitalize()} fitted "
            f"per cell and calendar month over {CALIBRATION[0]}-{CALIBRATION[1]} by "
            f"{FITTING}."
        ),
        "license": "CC-BY-4.0",
        "resource_type": "dataset",
        "keywords": ["drought", "SPEI", "water balance", "monthly", "gridded"],
        "contact": [
            {
                "organization": "Alliance of Bioversity International and CIAT",
                "roles": ["producer", "processor", "maintainer"],
                "url": "https://alliancebioversityciat.org/",
            }
        ],
        "related_publications": [
            {"doi": "10.1175/2009JCLI2909.1"},
            {"doi": "10.1002/joc.3887"},
        ],
        "created": today,
        "updated": today,
        "spatial": {
            "bbox": [grid.west, grid.south, grid.east, grid.north],
            "geography": ["world"],
            "crs": "EPSG:4326",
        },
        "temporal": {
            "start_date": f"{pins['months'][0]}-01",
            "end_date": str(
                (np.datetime64(pins["months"][1], "M") + np.timedelta64(1, "M")).astype(
                    "datetime64[D]"
                )
                - np.timedelta64(1, "D")
            ),
            "update_frequency": "monthly",
        },
        "structures": [
            {
                "name": "zarr",
                "dimensions": [
                    {
                        "name": "time",
                        "type": "temporal",
                        "step": "P1M",
                        "description": "Month; for SPEI, the last month of the window.",
                    },
                    {
                        "name": "cell",
                        "type": "xy",
                        "step": grid.res,
                        "unit": "degree",
                        "description": "Regular 0.05 degree grid (the CHIRPS v3 grid).",
                    },
                ],
                "variables": variables,
            }
        ],
        "processing": [
            {"id": "source", "description": "Daily inputs.", "derived_from": derived},
            {
                "id": "spei",
                "description": "Daily to monthly sums, water balance, window sums, "
                "distribution fit and standardisation.",
                "date": today,
                "code": {
                    "url": "https://github.com/CGIAR-Climate-Data-Hub/cdh-data-pipeline"
                },
            },
        ],
        "data": [
            {
                "name": "zarr",
                "locations": [
                    {
                        "url": f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{STORE_NAME}",
                        "title": "HTTPS",
                    }
                ],
                "processing_steps": ["spei"],
                "media_type": "application/vnd.zarr; version=3",
            }
        ],
    }
    path = Path(OUTPUT, f"{DATASET}.yaml")
    header = (
        "# yaml-language-server: $schema=https://cgiar-climate-data-hub.github.io/"
        "cdh-metadata-standard/v0.4.1/schemas/profiles/cdh.schema.json\n"
    )
    path.write_text(
        header + yaml.safe_dump(rec, sort_keys=False, width=100), encoding="utf-8"
    )
    log.info("wrote %s", path)


def update():
    """Add new months and recompute the last ``RECOMPUTE_MONTHS``, without refitting.

    Re-pins the inputs to their current revisions. Revisions inside the
    calibration period are not picked up here: rebuild ``spei`` with a fresh
    ``monthly_inputs`` for that (an explicit refit).
    """
    old = _inputs()
    old_months = _months(old)
    pins = _resolve_inputs()
    months = _months(pins)
    if months[0] != old_months[0]:
        raise SystemExit("the first month changed; rebuild instead of updating")
    start = max(len(old_months) - RECOMPUTE_MONTHS, 0)
    rows = slice(start, len(months))
    step = f"update-{pins['months'][1]}"
    _, area = _area()
    root = C.open_store_rw(URL)
    root.attrs["build_complete"] = False
    C.extend_time(root, months, WINDOWS)
    tl = C.TileLog(TILE_LOG)
    done = tl.done(step)
    src = _open(pins)
    glat, _ = CHIRPS_GRID.coords()
    log.info(
        "update: months %s..%s (recompute from %s)", *pins["months"], months[start]
    )
    for tile in C.tiles(area):
        if tile.key in done:
            continue
        if not tl.done("monthly_inputs").get(tile.key, {}).get("land_cells"):
            tl.mark(step, tile.key, {})
            continue
        precip, pet, _ = _monthly(src, months[rows], tile, glat[tile.row0 : tile.row1])
        ry, rx = _rows(tile, area)
        root["precip"][rows, ry, rx] = precip.astype("f4")
        root["pet"][rows, ry, rx] = (
            pet if pet is not None else precip * np.nan
        ).astype("f4")
        for w in WINDOWS:
            _tile_spei(root, tile, area, months, w, rows=rows)
        tl.mark(step, tile.key, {})
    root.attrs.update(
        time_coverage_end=str(months[-1]),
        last_updated=_today(),
    )
    history = old.get("history", []) + [{k: old[k] for k in ("resolved", "months")}]
    write_json(INPUTS, {**pins, "history": history})
    C.consolidate(URL)
    log.info("update done; run verify, then write_sources and write_card")


def publish_plan():
    """Report what ``publish`` would send to Hugging Face. Sends nothing."""
    log.info(
        "plan for %s: %s",
        HF_REPO,
        hf.publish(OUTPUT, HF_REPO, STORE_NAME, HF_SIDECARS, dry_run=True),
    )


def publish():
    """Send the files that differ from the Hugging Face repo, then confirm it matches."""
    pins = _inputs()
    result = hf.publish(
        OUTPUT,
        HF_REPO,
        STORE_NAME,
        HF_SIDECARS,
        message=f"{STORE_NAME}: coverage {pins['months'][0]} to {pins['months'][1]}",
    )
    log.info("published: %s", result)


if __name__ == "__main__":
    run(
        inspect,
        create_store,
        monthly_inputs,
        spei,
        verify,
        write_sources,
        write_card,
        write_cdh,
        on_request=(update, publish_plan, publish),
    )
