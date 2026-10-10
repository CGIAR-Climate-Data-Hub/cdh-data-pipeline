"""Small HDF5 fixtures exercise CHIRPS packing, manifests and incremental updates."""

import itertools
import json
import shutil
import urllib.request

import numpy as np
import pytest
import xarray as xr
import zarr

from cdh_data_pipeline import read_manifest, record
from recipes import chirps

VERSIONS = itertools.count()


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(chirps, "INPUT", str(source))
    monkeypatch.setattr(chirps, "OUTPUT", str(output))
    monkeypatch.setitem(chirps.ENCODING, "chunks", (3, 2, 2))
    monkeypatch.setitem(chirps.ENCODING, "shards", (3, 4, 4))
    return source, output / chirps.STORE_NAME


def name(start):
    return f"chirps-v3.0.rnl.{start[:4]}.days_p05.nc"


def write_source(path, start, values):
    """Write a 4x4 south-up file shaped like a CHC yearly NetCDF."""
    values = np.broadcast_to(
        np.asarray(values, dtype="float32")[:, None, None], (len(values), 4, 4)
    ).copy()
    # Source row 3 is the northernmost, so this is store pixel (0, 0).
    values[:, 3, 0] = -9999
    first = (np.datetime64(start) - np.datetime64("1980-01-01")).astype(int)
    ds = xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), values)},
        coords={
            "time": (
                "time",
                (first + np.arange(len(values))).astype("float64"),
                {"units": "days since 1980-1-1 0:0:0", "calendar": "gregorian"},
            ),
            "latitude": np.float32(20 + 0.05 * np.arange(4)),
            "longitude": np.float32(10 + 0.05 * np.arange(4)),
        },
    )
    ds.to_netcdf(path, engine="h5netcdf", encoding={"precip": {"_FillValue": -9999.0}})
    return path


def source_file(directory, start, values):
    """Cache a new upstream version of one year, as fetch would."""
    path = write_source(directory / name(start), start, values)
    version = {"size": path.stat().st_size, "modified": str(next(VERSIONS))}
    record(path, f"ftp://test/{path.name}", version)
    return path


def store_files(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def first_shard(files):
    return {p: b for p, b in files.items() if p.as_posix().startswith("precip/c/0/")}


def precip(root):
    with xr.open_zarr(root, consolidated=True) as ds:
        assert ds.attrs["build_complete"]
        return ds.precip[:, 1, 1].values


def test_build_then_update(recipe):
    source, root = recipe
    source_file(source, "2023-12-28", [0, 0.04, 0.06, 3276.7])
    chirps.build_zarr()
    before = store_files(root)
    chirps.build_zarr()
    assert store_files(root) == before, "an unchanged cache must not rewrite anything"
    # New days extend the store; the full first time shard (days 0-2) is untouched.
    source_file(source, "2024-01-01", [4, 5, 6])
    chirps.build_zarr()
    assert first_shard(store_files(root)) == first_shard(before)
    np.testing.assert_allclose(precip(root), [0, 0, 0.1, 3276.7, 4, 5, 6])
    with xr.open_zarr(root, consolidated=True) as ds:
        assert np.isnan(ds.precip[:, 0, 0]).all()
        assert ds.time.values[-1] == np.datetime64("2024-01-03")
        assert ds.precip.attrs["units"] == "mm/day"
        assert ds.precip.attrs["spatial:shape"] == [4, 4]
        assert ds.y.values[0] > ds.y.values[-1]
        assert set(ds.attrs["sources"]) == {name("2023"), name("2024")}
    published = json.loads((root.parent / "sources.json").read_text())
    assert published == ds.attrs["sources"]


def test_revised_file_rewrites_only_its_days(recipe):
    source, root = recipe
    source_file(source, "2023-12-28", [0, 1, 2, 3])
    source_file(source, "2024-01-01", [4, 5, 6])
    chirps.build_zarr()
    before = store_files(root)
    source_file(source, "2024-01-01", [7, 8, 9])
    chirps.build_zarr()
    assert first_shard(store_files(root)) == first_shard(before)
    np.testing.assert_allclose(precip(root), [0, 1, 2, 3, 7, 8, 9])
    source_file(source, "2023-12-28", [10, 11, 12, 13])
    chirps.build_zarr()
    np.testing.assert_allclose(precip(root), [10, 11, 12, 13, 7, 8, 9])


def test_fresh_cache_updates_from_changed_files_only(recipe, monkeypatch):
    source, root = recipe
    source_file(source, "2023-12-28", [0, 1, 2, 3])
    source_file(source, "2024-01-01", [4, 5, 6])
    chirps.build_zarr()
    # Another machine: same store, empty cache holding only the revised year.
    other = source.parent / "other"
    other.mkdir()
    monkeypatch.setattr(chirps, "INPUT", str(other))
    source_file(other, "2024-01-01", [7, 8, 9])
    chirps.build_zarr()
    np.testing.assert_allclose(precip(root), [0, 1, 2, 3, 7, 8, 9])


def test_fetch_downloads_only_versions_missing_from_the_store(recipe, monkeypatch):
    source, root = recipe
    server = source.parent / "server"
    server.mkdir()
    versions, downloads = {}, []

    def publish(start, values):
        path = write_source(server / name(start), start, values)
        versions[path.name] = {
            "size": path.stat().st_size,
            "modified": str(next(VERSIONS)),
        }

    def download(url, dest, version):
        downloads.append(dest.name)
        shutil.copy(server / dest.name, dest)
        record(dest, url, version)

    monkeypatch.setattr(chirps, "ftp_files", lambda url, suffix: dict(versions))
    monkeypatch.setattr(chirps, "download", download)
    publish("2023-12-28", [0, 1, 2, 3])
    publish("2024-01-01", [4, 5, 6])
    chirps.fetch()
    chirps.build_zarr()
    # Another machine with an empty cache downloads nothing until CHC revises a year.
    other = source.parent / "other"
    other.mkdir()
    monkeypatch.setattr(chirps, "INPUT", str(other))
    chirps.fetch()
    assert sorted(downloads) == [name("2023"), name("2024")]
    publish("2024-01-01", [7, 8, 9])
    chirps.fetch()
    assert downloads[2:] == [name("2024")]
    assert [p.name for p in other.glob("*.nc")] == [name("2024")]
    chirps.build_zarr()
    np.testing.assert_allclose(precip(root), [0, 1, 2, 3, 7, 8, 9])


def test_legacy_cache_is_adopted_without_downloading(recipe, monkeypatch):
    source, _ = recipe
    path = write_source(source / name("2023"), "2023-12-28", [0, 1, 2, 3])
    version = {"size": path.stat().st_size, "modified": "20260817232837"}
    monkeypatch.setattr(chirps, "ftp_files", lambda url, suffix: {path.name: version})
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a: pytest.fail("fetched"))
    chirps.fetch()
    assert read_manifest(source)[path.name]["version"] == version


def test_failed_update_resumes_next_run(recipe):
    source, root = recipe
    source_file(source, "2023-12-28", [0, 1, 2, 3])
    chirps.build_zarr()
    source_file(source, "2024-01-01", [4, 5, 4000])
    with pytest.raises(ValueError, match="range"):
        chirps.build_zarr()
    assert not zarr.open_group(root, use_consolidated=False).attrs["build_complete"]
    source_file(source, "2024-01-01", [4, 5, 6])
    chirps.build_zarr()
    np.testing.assert_allclose(precip(root), np.arange(7))


def test_time_gap_is_rejected_without_writing_data(recipe):
    source, root = recipe
    source_file(source, "2023-12-28", [0, 1, 2, 3])
    chirps.build_zarr()
    before = {p: b for p, b in store_files(root).items() if p.parts[0] == "precip"}
    source_file(source, "2024-01-02", [4, 5])
    with pytest.raises(ValueError, match="does not continue"):
        chirps.build_zarr()
    after = {p: b for p, b in store_files(root).items() if p.parts[0] == "precip"}
    assert after == before


def test_empty_cache_is_an_error(recipe):
    with pytest.raises(ValueError, match="no CHIRPS files"):
        chirps.build_zarr()


def test_layout_change_requires_deleting_the_store(recipe, monkeypatch):
    source, root = recipe
    source_file(source, "2023-12-28", [0, 1, 2, 3])
    chirps.build_zarr()
    monkeypatch.setitem(chirps.ENCODING, "shards", (6, 4, 4))
    with pytest.raises(ValueError, match="delete it to rebuild"):
        chirps.build_zarr()
    np.testing.assert_allclose(precip(root), [0, 1, 2, 3])


def test_failed_download_fails_fetch(recipe, monkeypatch):
    version = {"size": 1, "modified": "x"}
    monkeypatch.setattr(
        chirps, "ftp_files", lambda url, suffix: {name("2024"): version}
    )

    def download(*args, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(chirps, "download", download)
    with pytest.raises(OSError, match="connection reset"):
        chirps.fetch()
