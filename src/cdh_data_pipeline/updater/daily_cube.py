"""Keep a cube built by :mod:`cdh_data_pipeline.daily_cube` in step with its source.

A published cube drifts from its source two ways, and both are the same problem:
some day in the cube no longer matches what the server holds.

**new data**
    Days published since the last run. Beyond the cube's end they extend the time
    axis; inside it they fill a gap the build never had.
**reprocessed or repaired data**
    A day whose upstream content changed -- CHC republishes months, and ERA5T is
    replaced by ERA5 final weeks later -- or a day that is nodata locally because
    its source raster was corrupt.

Both are found the same way: list what the server has, compare each day's
signature against the one recorded at ingest, act on the difference. So the
updater never works from a calendar. Running it twice is a no-op; running it
after a reprocessing event re-ingests exactly the days that changed; and a day
left nodata by a bad file is never recorded, so it reappears as work until it is
actually written.

Where the state lives
---------------------
``<dataset prefix>/update-state.json``, beside the store and inside the dataset's
own prefix, written with the same ``write_json`` the recipes use for
``sources.json``. It is url-addressed, so it works against ``s3://`` unchanged,
and deleting the prefix deletes it with the data (CONVENTIONS.md).

Not the store's root attrs, which is where per-*file* provenance goes for
datasets built from a handful of yearly files. This state is per *day* per
variable: CHIRTS-ERA5 from 1981 is ~16,600 days x 2 variables. As a root attr
that is megabytes of JSON re-read on every ``open_group`` and duplicated by
``consolidate_metadata`` -- it would make the published cube slow to open for
every consumer, to record something only the updater reads.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import zarr

from cdh_data_pipeline.daily_cube import GridSpec, cube_store
from cdh_data_pipeline.recipe import log
from cdh_data_pipeline.storage import open_store, write_json
from cdh_data_pipeline.utils import MeteoVariable
from cdh_data_pipeline.zarr import EPOCH, open_for_write

from .remote import ArchiveRemote, RemoteFile

STATE_NAME = "update-state.json"


def _split_url(url: str) -> tuple[str, str]:
    """``(prefix, name)`` for a url, without ``pathlib`` touching the scheme."""
    prefix, _, name = url.rstrip("/").rpartition("/")
    return prefix, name


def read_json(url: str) -> dict:
    """Read a JSON document from a local path or object-store url; {} if absent."""
    prefix, name = _split_url(url)
    try:
        return json.loads(bytes(open_store(prefix).get(name).bytes()))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # obstore raises its own not-found type
        if type(exc).__name__ in {"NotFoundError", "FileNotFoundError"}:
            return {}
        raise


@dataclass
class CubePlan:
    """What an update would do, per variable, before it does it."""

    append: dict[str, list[RemoteFile]] = field(default_factory=dict)
    revise: dict[str, list[RemoteFile]] = field(default_factory=dict)
    unchanged: int = 0
    before_start: int = 0
    incomplete_months: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(self.append.values()) and not any(self.revise.values())

    def for_variable(self, key: str) -> list[RemoteFile]:
        return sorted(
            self.append.get(key, []) + self.revise.get(key, []), key=lambda r: r.day
        )

    def describe(self) -> str:
        parts = []
        for key in sorted(set(self.append) | set(self.revise)):
            new, rev = len(self.append.get(key, [])), len(self.revise.get(key, []))
            if not (new or rev):
                continue
            bits = []
            if new:
                d = self.append[key]
                bits.append(f"{new} new ({d[0].day} -> {d[-1].day})")
            if rev:
                bits.append(f"{rev} to re-ingest")
            parts.append(f"{key}: " + ", ".join(bits))
        if not parts:
            parts.append("nothing to do")
        if self.incomplete_months:
            parts.append(
                "skipping incomplete month(s): " + ", ".join(self.incomplete_months)
            )
        if self.before_start:
            parts.append(f"{self.before_start} day(s) precede the cube's start")
        return "; ".join(parts) + f" ({self.unchanged} already current)"


class CubeUpdater:
    """Keep a datacube in step with its source archive.

    Parameters
    ----------
    url
        The ``.zarr`` store url. State is written beside it, in the same prefix.
    require_complete_months
        Ingest a month only once every day of it is published. Right for CHIRPS,
        which drops a whole month at a time; wrong for CHIRTS, which trickles
        individual days at a ~5-day lag and would otherwise be held back by up to
        a month.
    verify
        Optional callable run before recording existing days, to confirm the cube
        holds what the caller thinks. Recording state writes down whatever it is
        told, so an unverified guess becomes a wrong label that later updates act
        on.
    """

    def __init__(
        self,
        url: str,
        remote: ArchiveRemote,
        variables: Sequence[MeteoVariable],
        grid: GridSpec,
        require_complete_months: bool = False,
        cache_dir=None,
        workers: int = 1,
        verify: Callable[[], None] | None = None,
        extra_state: Mapping | None = None,
    ):
        self.url = url.rstrip("/")
        self.remote = remote
        self.variables = list(variables)
        self.grid = grid
        self.require_complete_months = require_complete_months
        # A download cache is genuinely local, so Path is right here.
        self.cache_dir = Path(cache_dir or "input/_raster_cache")
        self.workers = workers
        self.verify = verify
        self.extra_state = dict(extra_state or {})
        self.prefix = _split_url(self.url)[0]
        self.state_url = f"{self.prefix}/{STATE_NAME}"

    # ----------------------------------------------------------------- state

    def load_state(self) -> dict:
        state = read_json(self.state_url)
        state.setdefault("days", {})
        return state

    def save_state(self, state: dict) -> None:
        write_json(self.state_url, state)

    # ------------------------------------------------------------- cube info

    def _root(self, mode: str = "r"):
        return zarr.open_group(cube_store(self.url), mode=mode, use_consolidated=False)

    def cube_span(self) -> tuple[date, date]:
        root = self._root()
        days = EPOCH + np.asarray(root["time"][:]).astype("timedelta64[D]")
        return days.min().astype(object), days.max().astype(object)

    def present_days(self, var: MeteoVariable) -> set[date]:
        """Dates this variable actually holds data for.

        Sampled coarsely: a day that was never written is uniformly fill, so any
        land pixel settles it. A day left nodata by a corrupt source file is
        therefore *not* present, and will be planned as work -- which is how a bad
        raster repairs itself on the next run.
        """
        root = self._root()
        days = EPOCH + np.asarray(root["time"][:]).astype("timedelta64[D]")
        sample = np.asarray(root[var.nc_name][:, ::101, ::401])
        ok = (sample != var.fill).any(axis=(1, 2))
        return {d.astype(object) for d, good in zip(days, ok) if good}

    def record_existing(self, progress: bool = True) -> dict:
        """Record upstream signatures for the days the cube already holds.

        Run once after building from a local archive. Without it every day looks
        un-ingested and the first update re-downloads the whole record. Reads
        directory listings only -- no rasters.
        """
        if self.verify is not None:
            self.verify()
        state = self.load_state()
        state.update(self.extra_state)
        days = state["days"]
        start, end = self.cube_span()
        now = datetime.now().isoformat(timespec="seconds")
        totals = {}
        for var in self.variables:
            present = self.present_days(var)
            bucket = days.setdefault(var.key, {})
            for year in range(start.year, end.year + 1):
                for r in self.remote.list_year(var.key, year):
                    if start <= r.day <= end and r.day in present:
                        bucket[r.day.isoformat()] = r.signature
            totals[var.key] = len(bucket)
            if progress:
                log.info("  %s: recorded %d day(s)", var.key, len(bucket))
        state["recorded"] = now
        self.save_state(state)
        return {"recorded": totals, "cube_span": [str(start), str(end)]}

    # ------------------------------------------------------------------ plan

    def plan(self, through: date | None = None) -> CubePlan:
        seen = self.load_state()["days"]
        cube_start, cube_end = self.cube_span()
        today = through or date.today()
        plan = CubePlan()

        for var in self.variables:
            bucket = seen.get(var.key, {})
            new, rev = [], []
            for year in range(cube_start.year, today.year + 1):
                rows = self.remote.list_year(var.key, year)
                if not rows:
                    continue
                if self.require_complete_months:
                    done = self.remote.complete_months(var.key, year)
                    months = {(year, r.day.month) for r in rows}
                    for ym in sorted(months - done):
                        label = f"{ym[0]}-{ym[1]:02d}"
                        if label not in plan.incomplete_months:
                            plan.incomplete_months.append(label)
                    rows = [r for r in rows if (year, r.day.month) in done]
                for r in rows:
                    if r.day > today:
                        continue
                    if r.day < cube_start:
                        plan.before_start += 1
                        continue
                    if bucket.get(r.day.isoformat()) == r.signature:
                        plan.unchanged += 1
                    elif r.day > cube_end:
                        new.append(r)  # extends the time axis
                    else:
                        # Inside the span: reprocessed upstream, or a day the
                        # build left nodata. Both are in-place writes.
                        rev.append(r)
            plan.append[var.key] = sorted(new, key=lambda r: r.day)
            plan.revise[var.key] = sorted(rev, key=lambda r: r.day)
        return plan

    # ----------------------------------------------------------------- apply

    def apply(
        self, plan: CubePlan, keep_downloads: bool = False, progress: bool = True
    ) -> dict:
        import rasterio

        if plan.empty:
            return {"written": 0, "seconds": 0.0}

        state = self.load_state()
        state.update(self.extra_state)
        store = cube_store(self.url)
        root = open_for_write(store)
        # A reader that finds this false is looking at a store mid-write.
        root.attrs["build_complete"] = False

        # Grow the axis first, so every write lands on a real index.
        newest = max(r.day for v in self.variables for r in plan.for_variable(v.key))
        axis = self._axis(root)
        if newest > axis[-1]:
            n = (
                np.datetime64(newest, "D") - np.datetime64(axis[0], "D")
            ).astype(int) + 1
            stored = np.asarray(root["time"][:])
            for var in self.variables:
                a = root[var.nc_name]
                a.resize((n, a.shape[1], a.shape[2]))
            root["time"].resize((n,))
            s0 = int(stored[0])
            root["time"][len(stored) : n] = np.arange(
                s0 + len(stored), s0 + n, dtype="int32"
            )
            axis = self._axis(root)
            if progress:
                log.info("  axis extended to %s", axis[-1])

        index = {d: i for i, d in enumerate(axis)}
        t0 = _time.time()
        written = 0
        failed: list[str] = []

        for var in self.variables:
            todo = plan.for_variable(var.key)
            if not todo:
                continue
            bucket = state["days"].setdefault(var.key, {})
            for run in _contiguous(todo):
                if progress:
                    log.info(
                        "  %s: %s -> %s (%d day(s))",
                        var.key, run[0].day, run[-1].day, len(run),
                    )
                paths = self.remote.download_many(
                    run, self.cache_dir, workers=self.workers, progress=progress
                )
                block = np.full(
                    (len(run), self.grid.height, self.grid.width),
                    var.fill,
                    dtype=var.dtype,
                )
                good = []
                # paths is positional: a failed download is None at its own index,
                # so a day never takes another day's raster.
                for i, (r, p) in enumerate(zip(run, paths)):
                    if p is None:
                        failed.append(f"{var.key} {r.day}: download failed")
                        continue
                    try:
                        with rasterio.open(p) as src:
                            if (src.height, src.width) != (
                                self.grid.height,
                                self.grid.width,
                            ):
                                raise ValueError(f"grid {src.height}x{src.width}")
                            arr = src.read(1).astype(np.float32, copy=False)
                    except Exception as exc:  # noqa: BLE001  reported below
                        failed.append(
                            f"{var.key} {r.day}: {type(exc).__name__}: {str(exc)[:60]}"
                        )
                        continue
                    arr[arr == self.grid.nodata] = np.nan
                    block[i] = var.encode(arr)
                    good.append(r)

                i0 = index[run[0].day]
                root[var.nc_name][i0 : i0 + len(run)] = block
                del block
                written += len(good)
                # Only record what actually landed: a day whose download was
                # corrupt must stay unrecorded so the next run retries it.
                for r in good:
                    bucket[r.day.isoformat()] = r.signature
                self.save_state(state)  # checkpoint per run
                if not keep_downloads:
                    for p in paths:
                        if p is not None:
                            p.unlink(missing_ok=True)

        axis = self._axis(root)
        now = datetime.now().isoformat(timespec="seconds")
        root.attrs["time_coverage_start"] = str(axis[0])
        root.attrs["time_coverage_end"] = str(axis[-1])
        root.attrs["last_updated"] = now
        root.attrs["build_complete"] = True
        zarr.consolidate_metadata(store)
        state["updated"] = now
        self.save_state(state)

        out = {
            "written": written,
            "seconds": round(_time.time() - t0, 1),
            "coverage": [str(axis[0]), str(axis[-1])],
        }
        if failed:
            out["failed"] = failed
            if progress:
                log.warning(
                    "%d day(s) could not be read and stay nodata; "
                    "they will be retried next run", len(failed),
                )
                for f in failed[:10]:
                    log.warning("    %s", f)
        return out

    def update(
        self,
        through: date | None = None,
        dry_run: bool = False,
        keep_downloads: bool = False,
        progress: bool = True,
    ) -> dict:
        plan = self.plan(through=through)
        if progress:
            log.info("plan: %s", plan.describe())
        if dry_run or plan.empty:
            return {"plan": plan.describe(), "written": 0, "dry_run": dry_run}
        result = self.apply(
            plan, keep_downloads=keep_downloads, progress=progress
        )
        result["plan"] = plan.describe()
        return result

    # -------------------------------------------------------------- plumbing

    @staticmethod
    def _axis(root) -> list[date]:
        return (
            (EPOCH + np.asarray(root["time"][:]).astype("timedelta64[D]"))
            .astype(object)
            .tolist()
        )


def _contiguous(files: Sequence[RemoteFile]) -> list[list[RemoteFile]]:
    """Split a date-sorted list into runs of consecutive calendar days."""
    runs: list[list[RemoteFile]] = []
    for r in files:
        if runs and r.day - runs[-1][-1].day == timedelta(days=1):
            runs[-1].append(r)
        else:
            runs.append([r])
    return runs
