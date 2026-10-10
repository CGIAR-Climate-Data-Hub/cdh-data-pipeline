"""One builder for any dated-raster archive -> daily Zarr v3 datacube.

CHIRPS and CHIRTS differ only in grid, variables and editorial metadata; the
conversion itself is the same job. That job lives here, so a fix or a measurement
applies to every mission at once.

What the defaults encode, all of it measured rather than assumed:

* **Plain Zstandard, not Blosc.** Blosc is ~15% smaller on sparse zero-heavy
  fields like precipitation and marginally *larger* on dense ones like
  temperature -- but GDAL ships without Blosc in most builds, so a Blosc store is
  unreadable from QGIS, R's terra/stars, and every other GDAL tool.
* **No sharding.** Sharding cuts file count but *raises* request count (the
  reader range-reads each chunk and the shard index on top), and GDAL cannot read
  it at any version: ``Unsupported codec: sharding_indexed``.
* **Chunks of (128, 300, 300).** The chunk is the unit of one HTTP range request.
  At 100 px a 45-year country query cost ~1,433 requests against a
  5,000-per-5-minutes budget; at 300 px it costs ~260, for the same bytes.

Reading is band-wise: a block of ``(time_chunk, lat_chunk, full width)`` is held
at once, about 550 MB, rather than a whole time slab of several GB. Source
GeoTIFFs are striped one row per block, so a row-range window is cheap.

Destinations are **url strings**, not ``Path``s (see CONVENTIONS.md): ``pathlib``
flattens ``s3://bucket`` to ``s3:/bucket``. ``Path`` is used only for the local
raster archive and download cache, which are genuinely local.
"""

from __future__ import annotations

import glob
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import zarr
from zarr.storage import ObjectStore

from cdh_data_pipeline.storage import clear_store, open_store
from cdh_data_pipeline.utils import (
    MeteoVariable,
    dense_daily_axis,
    parse_dotted_date,
)
from cdh_data_pipeline.zarr import (
    EPOCH,
    TIME_UNITS,
    CubeGeometry,
    compressors,
    open_for_write,
)

#: Chunks for request count, not for bytes; see the module docstring.
DEFAULT_GEOMETRY = CubeGeometry(
    time_chunk=128, lat_chunk=300, lon_chunk=300, sharded=False
)
DEFAULT_CODEC = "zstd"


def cube_store(url: str) -> ObjectStore:
    """Zarr store handle for a local path or object-store url."""
    return ObjectStore(open_store(url))


def store_stats(url: str) -> tuple[int, int]:
    """Total bytes and object count under ``url``.

    Listed through obstore rather than walked with ``Path.rglob``, so the figure
    is the same whether the store is on disk or in a bucket.
    """
    total = count = 0
    for batch in open_store(url).list():
        for meta in batch:
            total += meta["size"]
            count += 1
    return total, count


@dataclass(frozen=True)
class GridSpec:
    """The fixed grid a mission's rasters are published on."""

    height: int
    width: int
    res: float
    north: float
    west: float = -180.0
    #: Sentinel in the source rasters. Both CHC products leave it undeclared in
    #: the GeoTIFF profile, so it cannot be read from the file.
    nodata: float = -9999.0
    crs: str = "EPSG:4326"

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre lat/lon, built from the nominal resolution.

        Not taken from the GeoTIFF transform: the files carry a float32 round-trip
        of the spacing, which drifts a fraction of a pixel across the width and
        lands bbox lookups on the wrong column at the east edge.
        """
        lat = np.round(self.north - self.res * (np.arange(self.height) + 0.5), 6)
        lon = np.round(self.west + self.res * (np.arange(self.width) + 0.5), 6)
        return lat, lon

    @property
    def south(self) -> float:
        return round(self.north - self.res * self.height, 6)

    @property
    def east(self) -> float:
        return round(self.west + self.res * self.width, 6)


@dataclass
class VariableScan:
    """Local files found for one variable, keyed by date."""

    variable: MeteoVariable
    files: dict[date, Path] = field(default_factory=dict)
    skipped_out_of_range: int = 0

    @property
    def dates(self) -> list[date]:
        return sorted(self.files)


class RasterCubeBuilder:
    """Convert a local archive of dated rasters into a Zarr v3 datacube.

    Expects ``<root>/<variable folder>/<year>/*.tif``; a variable whose ``folder``
    is ``""`` is read straight from ``<root>/<year>/``.

    Parameters
    ----------
    variables
        One :class:`MeteoVariable` per array to write. Each carries its own dtype,
        scale and fill, so a cube can hold variables that could not share an
        encoding.
    init_year, end_year
        Bounds of the **time axis**, not merely a file filter. A cube can be
        allocated for its final span while only part of the archive exists; the
        empty years cost almost nothing, since Zarr never writes a chunk that is
        uniformly fill, and :meth:`ingest` fills them in later. Without that
        foresight, adding earlier years means a full rebuild -- a Zarr time axis
        only grows forward.
    """

    def __init__(
        self,
        input_dir,
        grid: GridSpec,
        variables: Sequence[MeteoVariable],
        geometry: CubeGeometry | None = None,
        codec: str = DEFAULT_CODEC,
        init_year: int | None = None,
        end_year: int | None = None,
        workers: int = 6,
        pattern: str = "*.tif",
        global_attrs: Mapping | None = None,
    ):
        if init_year is not None and end_year is not None and init_year > end_year:
            raise ValueError(f"init_year ({init_year}) is after end_year ({end_year})")
        self.input_dir = Path(input_dir)
        self.grid = grid
        self.variables = list(variables)
        self.geometry = geometry or DEFAULT_GEOMETRY
        self.codec = codec
        self.init_year = init_year
        self.end_year = end_year
        self.workers = workers
        self.pattern = pattern
        self.global_attrs = dict(global_attrs or {})
        #: Rasters that failed to read during the last run/ingest.
        self.unreadable: list[str] = []
        self._unreadable_seen: set[str] = set()

    # ------------------------------------------------------------- scanning

    @property
    def year_range_label(self) -> str:
        lo = self.init_year if self.init_year is not None else "earliest"
        hi = self.end_year if self.end_year is not None else "latest"
        return f"{lo}..{hi}"

    def _in_range(self, day: date) -> bool:
        if self.init_year is not None and day.year < self.init_year:
            return False
        if self.end_year is not None and day.year > self.end_year:
            return False
        return True

    def _folder(self, var: MeteoVariable) -> Path:
        """Directory holding one variable, tolerating archive naming variants.

        A variable's ``folder`` is the canonical name, but archives label the same
        thing differently -- ``tmax_daily`` on one copy, ``tmax_daily_1981_2026``
        on another. An unambiguous prefix match is accepted so the caller does not
        have to rename directories on a read-only share; two matches are an error
        rather than a guess.
        """
        if not var.folder:
            return self.input_dir
        exact = self.input_dir / var.folder
        if exact.is_dir():
            return exact
        matches = sorted(d for d in self.input_dir.glob(f"{var.folder}*") if d.is_dir())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"{var.key!r}: {len(matches)} directories under {self.input_dir} "
                f"start with {var.folder!r} ({', '.join(m.name for m in matches)}); "
                f"rename or point at one of them directly"
            )
        return exact  # absent: let scan() report it with context

    def scan(self) -> dict[str, VariableScan]:
        out: dict[str, VariableScan] = {}
        for var in self.variables:
            scan = VariableScan(variable=var)
            folder = self._folder(var)
            found = 0
            for p in glob.glob(str(folder / "**" / self.pattern), recursive=True):
                try:
                    day = parse_dotted_date(p)
                except ValueError:
                    continue
                found += 1
                if not self._in_range(day):
                    scan.skipped_out_of_range += 1
                    continue
                scan.files[day] = Path(p)
            if not scan.files:
                # Distinguish "nothing on disk" from "nothing in the window"; they
                # need different fixes.
                if found:
                    raise FileNotFoundError(
                        f"{var.key!r}: {found} file(s) under {folder}, but none "
                        f"fall in {self.year_range_label}"
                    )
                raise FileNotFoundError(f"no rasters for {var.key!r} under {folder}")
            out[var.key] = scan
        return out

    def axis(self, scans: Mapping[str, VariableScan]) -> np.ndarray:
        """Dense daily axis: the union of coverage, bounded by init/end year."""
        every = [d for s in scans.values() for d in s.files]
        start = date(self.init_year, 1, 1) if self.init_year else min(every)
        end = date(self.end_year, 12, 31) if self.end_year else max(every)
        if start > min(every):
            raise ValueError(
                f"init_year {self.init_year} starts after the earliest file "
                f"({min(every)})"
            )
        return dense_daily_axis(start, end)

    def coverage_report(self, scans: Mapping[str, VariableScan]) -> list[str]:
        ax = self.axis(scans)
        notes = []
        for key, scan in scans.items():
            have = set(scan.files)
            gaps = sum(1 for d in ax.tolist() if d not in have)
            line = f"{key}: {len(have)} day(s) {scan.dates[0]} -> {scan.dates[-1]}"
            if gaps:
                line += f", {gaps} gap(s) on the shared axis"
            if scan.skipped_out_of_range:
                line += (
                    f", {scan.skipped_out_of_range} skipped outside "
                    f"{self.year_range_label}"
                )
            notes.append(line)
        return notes

    # ------------------------------------------------------------ integrity

    def verify_archive(self, quick: bool = False, progress: bool = True) -> dict:
        """Check every raster opens, decodes, and has the expected grid.

        The decode is not optional. Opening alone catches a truncated file (a
        TIFF's directory sits at the end), but not one whose directory is intact
        while its compressed strips are damaged -- seven such rasters in a CHIRTS
        year opened perfectly and failed only at ``read()``. ``quick=True`` skips
        the decode and will miss exactly that class.
        """
        import rasterio

        files: list[Path] = []
        for var in self.variables:
            files += [
                Path(p)
                for p in glob.glob(
                    str(self._folder(var) / "**" / self.pattern), recursive=True
                )
            ]

        def check(f: Path):
            try:
                with rasterio.open(f) as s:
                    if (s.height, s.width) != (self.grid.height, self.grid.width):
                        return f, f"grid {s.height}x{s.width}"
                    if not quick:
                        s.read(1)
                return None
            except Exception as exc:
                return f, f"{type(exc).__name__}: {str(exc)[:80]}"

        bad = []
        with ThreadPoolExecutor(self.workers) as ex:
            for r in ex.map(check, files):
                if r:
                    bad.append(r)
        if progress:
            print(f"  checked {len(files)} file(s); {len(bad)} unreadable")
            for f, why in bad[:20]:
                print(f"    {f.name}: {why}")
        return {
            "checked": len(files),
            "unreadable": [{"file": str(f), "reason": why} for f, why in bad],
        }

    # ---------------------------------------------------------------- write

    def _read_band(
        self,
        var: MeteoVariable,
        files: Sequence[Path | None],
        row0: int,
        nrows: int,
    ) -> np.ndarray:
        """Decode a (time, nrows, width) block of stored integers."""
        import rasterio
        from rasterio.windows import Window

        out = np.full((len(files), nrows, self.grid.width), var.fill, dtype=var.dtype)
        window = Window(0, row0, self.grid.width, nrows)

        def one(item):
            i, f = item
            if f is None:
                return
            try:
                with rasterio.open(f) as src:
                    arr = src.read(1, window=window).astype(np.float32, copy=False)
            except Exception as exc:
                # One bad file must not abandon a build that may be hours in. The
                # day stays nodata and is reported, so it can be refetched and
                # filled later with ingest().
                #
                # Deduplicated: the same file is reopened once per latitude band,
                # so a single corrupt raster would otherwise be listed
                # height/lat_chunk times and inflate the count.
                msg = (
                    f"{var.key} {parse_dotted_date(f)}: "
                    f"{type(exc).__name__}: {str(exc)[:70]}"
                )
                if msg not in self._unreadable_seen:
                    self._unreadable_seen.add(msg)
                    self.unreadable.append(msg)
                return
            arr[arr == self.grid.nodata] = np.nan
            out[i] = var.encode(arr)

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(one, enumerate(files)))
        return out

    def _write(self, arrays, ax, scans, only_days=None, progress=True) -> None:
        geom = self.geometry
        n_t = len(ax)
        blocks = [
            (t0, r0)
            for t0 in range(0, n_t, geom.time_chunk)
            for r0 in range(0, self.grid.height, geom.lat_chunk)
        ]
        if only_days is not None:
            blocks = [
                (t, r)
                for t, r in blocks
                if any(d in only_days for d in ax[t : t + geom.time_chunk].tolist())
            ]
        peak = geom.time_chunk * geom.lat_chunk * self.grid.width * 2 / 1e9
        if progress:
            print(f"  {len(blocks)} block(s), ~{peak:.2f} GB peak per variable")

        t_start = _time.time()
        for k, (t0, r0) in enumerate(blocks, 1):
            t1 = min(t0 + geom.time_chunk, n_t)
            nrows = min(geom.lat_chunk, self.grid.height - r0)
            days = ax[t0:t1].tolist()
            for var in self.variables:
                src = scans[var.key].files
                wanted = [
                    src.get(d) if (only_days is None or d in only_days) else None
                    for d in days
                ]
                if not any(w is not None for w in wanted):
                    continue
                block = self._read_band(var, wanted, r0, nrows)
                if only_days is not None:
                    # Partial pass: keep what is already stored for days this pass
                    # is not touching, or writing would erase them.
                    prior = np.asarray(arrays[var.key][t0:t1, r0 : r0 + nrows, :])
                    keep = np.array([d not in only_days for d in days])
                    block[keep] = prior[keep]
                arrays[var.key][t0:t1, r0 : r0 + nrows, :] = block
                del block
            if progress:
                el = _time.time() - t_start
                print(
                    f"  block {k}/{len(blocks)} ({days[0]}..{days[-1]} rows {r0}) "
                    f"[{el / k:.0f}s/block, "
                    f"eta {el / k * (len(blocks) - k) / 60:.1f} min]",
                    flush=True,
                )

    def run(self, url: str, overwrite: bool = True, progress: bool = True) -> dict:
        """Build the cube from scratch at ``url``.

        ``build_complete`` is false for the duration: a reader that finds it false
        is looking at a store mid-write and should not trust it.
        """
        from zarr.codecs import BytesCodec

        self.unreadable = []
        self._unreadable_seen = set()
        scans = self.scan()
        ax = self.axis(scans)
        lat, lon = self.grid.coords()
        self.geometry.validate(self.grid.height, self.grid.width)

        if progress:
            for line in self.coverage_report(scans):
                print(f"  {line}")
            print(f"  {len(ax)} day(s) x {len(self.variables)} variable(s)")

        store = cube_store(url)
        if overwrite:
            if not url.rstrip("/").endswith(".zarr"):
                raise ValueError(f"refusing to overwrite non-.zarr store: {url}")
            clear_store(open_store(url))
        root = zarr.create_group(store=store, overwrite=overwrite)
        root.attrs["build_complete"] = False
        self._write_coords(root, ax, lat, lon)
        arrays = {}
        for var in self.variables:
            a = root.create_array(
                name=var.nc_name,
                shape=(len(ax), self.grid.height, self.grid.width),
                chunks=self.geometry.chunks,
                shards=self.geometry.shards,
                dtype=var.dtype,
                fill_value=var.fill,
                serializer=BytesCodec(),
                compressors=compressors(self.codec),
                dimension_names=("time", "lat", "lon"),
            )
            a.attrs.update(var.zarr_attrs())
            arrays[var.key] = a

        t0 = _time.time()
        self._write(arrays, ax, scans, progress=progress)
        root.attrs.update(self._attrs(ax))
        root.attrs["build_complete"] = True
        zarr.consolidate_metadata(store)

        total, n_files = store_stats(url)
        if progress:
            print(
                f"\n  {total / 1e9:.2f} GB / {n_files:,} objects, "
                f"{total / len(ax) / 1e6:.2f} MB/day across "
                f"{len(self.variables)} variable(s) in {(_time.time() - t0) / 60:.1f} min"
            )
            if self.unreadable:
                print(
                    f"\n  {len(self.unreadable)} UNREADABLE file(s) -- those days "
                    f"are nodata; refetch and re-run the ingest step:"
                )
                for u in self.unreadable[:10]:
                    print(f"    {u}")
        return {
            "days": len(ax),
            "variables": [v.key for v in self.variables],
            "bytes": total,
            "files": n_files,
            "unreadable": list(self.unreadable),
        }

    def ingest(self, url: str, progress: bool = True) -> dict:
        """Write newly available days into an existing cube, without rebuilding."""
        store = cube_store(url)
        self.unreadable = []
        self._unreadable_seen = set()
        scans = self.scan()
        root = open_for_write(store)
        root.attrs["build_complete"] = False
        stored = np.asarray(root["time"][:])
        axis_days = (EPOCH + stored.astype("timedelta64[D]")).astype(object).tolist()
        first, last = axis_days[0], axis_days[-1]

        have = sorted({d for s in scans.values() for d in s.files})
        before = [d for d in have if d < first]
        beyond = [d for d in have if d > last]

        if beyond:
            n = (
                np.datetime64(max(beyond), "D") - np.datetime64(first, "D")
            ).astype(int) + 1
            for var in self.variables:
                a = root[var.nc_name]
                a.resize((n, a.shape[1], a.shape[2]))
            root["time"].resize((n,))
            s0 = int(stored[0])
            root["time"][len(stored) : n] = np.arange(
                s0 + len(stored), s0 + n, dtype="int32"
            )
            stored = np.asarray(root["time"][:])
            axis_days = (
                (EPOCH + stored.astype("timedelta64[D]")).astype(object).tolist()
            )
            if progress:
                print(f"  axis extended to {axis_days[-1]}")

        writable = {d for d in have if d >= first}
        if before and progress:
            print(
                f"  WARNING: {len(before)} day(s) precede the cube's start ({first}); "
                f"rebuild with init_year={min(before).year} to include them"
            )
        if not writable:
            root.attrs["build_complete"] = True
            zarr.consolidate_metadata(store)
            return {"written": 0, "before_start": len(before)}

        arrays = {v.key: root[v.nc_name] for v in self.variables}
        self._write(
            arrays,
            np.array(axis_days, dtype="datetime64[D]"),
            scans,
            only_days=writable,
            progress=progress,
        )
        root.attrs["time_coverage_start"] = str(axis_days[0])
        root.attrs["time_coverage_end"] = str(axis_days[-1])
        root.attrs["build_complete"] = True
        zarr.consolidate_metadata(store)
        return {
            "written": len(writable),
            "before_start": len(before),
            "coverage": [str(axis_days[0]), str(axis_days[-1])],
            "unreadable": list(self.unreadable),
        }

    # ------------------------------------------------------------ plumbing

    def _attrs(self, ax) -> dict:
        g = self.grid
        return {
            "Conventions": "CF-1.10",
            "spatial_resolution_deg": g.res,
            "geospatial_lat_min": g.south,
            "geospatial_lat_max": g.north,
            "geospatial_lon_min": g.west,
            "geospatial_lon_max": g.east,
            "time_coverage_start": str(ax[0]),
            "time_coverage_end": str(ax[-1]),
            "variables": [v.nc_name for v in self.variables],
            "build_year_range": self.year_range_label,
            **self.global_attrs,
        }

    def _write_coords(self, root, ax, lat, lon) -> None:
        days = (ax - EPOCH).astype("int32")
        t = root.create_array(
            name="time",
            shape=days.shape,
            chunks=days.shape,
            dtype="int32",
            dimension_names=("time",),
        )
        t[:] = days
        t.attrs.update(
            units=TIME_UNITS,
            calendar="proleptic_gregorian",
            standard_name="time",
            axis="T",
        )
        for nm, vals, unit, std, axis_ in (
            ("lat", lat, "degrees_north", "latitude", "Y"),
            ("lon", lon, "degrees_east", "longitude", "X"),
        ):
            a = root.create_array(
                name=nm,
                shape=vals.shape,
                chunks=vals.shape,
                dtype="float64",
                dimension_names=(nm,),
            )
            a[:] = vals
            a.attrs.update(units=unit, standard_name=std, axis=axis_)
        crs = root.create_array(name="spatial_ref", shape=(), chunks=(), dtype="int32")
        crs[...] = 0
        crs.attrs.update(
            {
                "grid_mapping_name": "latitude_longitude",
                "crs_wkt": self.grid.crs,
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            }
        )
