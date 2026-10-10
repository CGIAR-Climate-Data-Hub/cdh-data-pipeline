"""download() keeps a manifest of what each input cache holds and where it came from."""

import ftplib
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from cdh_data_pipeline import download, ftp_files, read_manifest


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "server"
    root.mkdir()

    def publish(name, content):
        (root / name).write_bytes(content)
        return (root / name).as_uri()

    return publish


def test_download_records_source_size_and_hash(server, tmp_path):
    url = server("a.bin", b"hello")
    dest = download(url, tmp_path / "cache" / "a.bin", version={"modified": "1"})
    entry = read_manifest(dest.parent)["a.bin"]
    assert dest.read_bytes() == b"hello"
    assert entry["url"] == url and entry["size"] == 5
    assert entry["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert entry["version"] == {"modified": "1"} and "retrieved" in entry


def test_current_copy_is_kept_and_new_version_refetched(server, tmp_path):
    url = server("a.bin", b"v1")
    dest = download(url, tmp_path / "a.bin", version="1")
    server("a.bin", b"v2")
    download(url, dest, version="1")
    assert dest.read_bytes() == b"v1"
    download(url, dest, version="2")
    assert dest.read_bytes() == b"v2"
    assert read_manifest(tmp_path)["a.bin"]["version"] == "2"


def test_files_cached_before_manifests_are_adopted_if_sizes_match(server, tmp_path):
    url = server("a.bin", b"new!")
    (tmp_path / "a.bin").write_bytes(b"old!")
    download(url, tmp_path / "a.bin", version={"size": 4})
    assert (tmp_path / "a.bin").read_bytes() == b"old!"
    (tmp_path / "b.bin").write_bytes(b"short")
    download(server("b.bin", b"longer"), tmp_path / "b.bin", version={"size": 6})
    assert (tmp_path / "b.bin").read_bytes() == b"longer"


def test_lazy_url_is_only_resolved_when_downloading(server, tmp_path):
    url = server("a.bin", b"x")
    download(lambda: url, tmp_path / "a.bin", version="1", source="stable")
    assert read_manifest(tmp_path)["a.bin"]["url"] == "stable"
    download(lambda: pytest.fail("resolved"), tmp_path / "a.bin", version="1")


def test_concurrent_downloads_keep_every_manifest_entry(server, tmp_path):
    urls = {f"{i}.bin": server(f"{i}.bin", bytes([i])) for i in range(16)}
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda kv: download(kv[1], tmp_path / kv[0]), urls.items()))
    assert sorted(read_manifest(tmp_path)) == sorted(urls)


def test_ftp_files_lists_size_and_modification_time(monkeypatch):
    files = {"/data/a.nc": (10, "20260817232837"), "/data/b.txt": (5, "20250101000000")}

    class FakeFTP:
        def __init__(self, host, timeout):
            assert host == "ftp.example.org"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def login(self):
            pass

        def voidcmd(self, cmd):
            return f"213 {files[cmd.split()[1]][1]}" if cmd.startswith("MDTM") else ""

        def nlst(self, path):
            assert path == "/data"
            return list(files)

        def size(self, path):
            return files[path][0]

    monkeypatch.setattr(ftplib, "FTP", FakeFTP)
    listing = ftp_files("ftp://ftp.example.org/data", ".nc")
    assert listing == {"a.nc": {"size": 10, "modified": "20260817232837"}}
