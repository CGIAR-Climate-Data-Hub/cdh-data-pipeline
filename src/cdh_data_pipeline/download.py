"""Download source files into a local input cache, recording what was fetched.

Each cache directory keeps a ``manifest.json``: per file, the source URL, size,
SHA-256, retrieval time and any upstream version (e.g. FTP size + modification
time, HTTP ETag). Recipes publish it as ``sources.json`` next to their outputs.
"""

import ftplib
import hashlib
import json
import os
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from cdh_data_pipeline.recipe import log

HARVARD = "https://dataverse.harvard.edu"
UA = {"User-Agent": "cdh-data-pipeline"}  # Dataverse blocks urllib's default
MANIFEST = "manifest.json"
_manifest_lock = threading.Lock()  # recipes download with thread pools


def read_manifest(directory):
    """Return ``{filename: entry}`` from ``directory``'s manifest, or ``{}``."""
    path = Path(directory, MANIFEST)
    return json.loads(path.read_text()) if path.exists() else {}


def write_manifest(directory, manifest):
    """Replace the manifest atomically, so an interrupted write keeps the old one."""
    path = Path(directory, MANIFEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    part.replace(path)


def record(dest, url, version=None, **extra):
    """Hash ``dest`` and add it to its directory's manifest. Returns the entry."""
    dest = Path(dest)
    with open(dest, "rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    stat = dest.stat()
    entry = {
        "url": url,
        "size": stat.st_size,
        "sha256": sha256,
        "retrieved": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(
            timespec="seconds"
        ),
        **({"version": version} if version is not None else {}),
        **extra,
    }
    with _manifest_lock:
        manifest = read_manifest(dest.parent)
        manifest[dest.name] = entry
        write_manifest(dest.parent, manifest)
    return entry


def ftp_files(url, suffix=""):
    """Return ``{filename: {"size", "modified"}}`` for files in an ftp:// directory.

    Uses anonymous login. ``modified`` is the server's MDTM timestamp, so each value
    works as ``download(..., version=...)``: a changed file is fetched again.
    """
    parts = urllib.parse.urlsplit(url)
    with ftplib.FTP(parts.hostname, timeout=60) as ftp:
        ftp.login()
        ftp.voidcmd("TYPE I")  # some servers refuse SIZE in ASCII mode
        paths = [p for p in ftp.nlst(parts.path) if p.endswith(suffix)]
        return {
            Path(p).name: {
                "size": ftp.size(p),
                "modified": ftp.voidcmd(f"MDTM {p}")[4:],
            }
            for p in paths
        }


def download(url, dest, *, version=None, source=None):
    """Download ``url`` to ``dest`` unless the cached copy is current. Returns ``dest``.

    ``version`` identifies the upstream file (e.g. ``{"size": ..., "modified": ...}``);
    a cached copy recorded with a different version is downloaded again. A file
    cached before manifests existed is adopted if its size matches ``version``.
    ``url`` may be a function returning the URL, called only if a download is
    needed; then pass ``source``, the stable URL to record instead.
    """
    dest = Path(dest)
    entry = read_manifest(dest.parent).get(dest.name)
    if dest.exists():
        if entry is not None and entry.get("version") == version:
            return dest
        if entry is None and (version or {}).get("size") in (None, dest.stat().st_size):
            log.info("recording cached %s", dest.name)
            record(dest, source or url, version)
            return dest
        log.info("%s changed upstream, re-downloading", dest.name)
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")  # so a partial file never looks done
    log.info("downloading %s", dest.name)
    if callable(url):
        url = url()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
        headers = {
            k.lower().replace("-", "_"): r.headers[k]
            for k in ("ETag", "Last-Modified")
            if r.headers.get(k)
        }
    part.replace(dest)
    record(dest, source or url, version, **headers)
    log.info("downloaded %s (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_dataverse(doi, filenames, dest_dir, *, version=":latest", server=HARVARD):
    """Download ``filenames`` from a Dataverse dataset into ``dest_dir``.

    Skips files already cached at this version. Needs ``DATAVERSE_TOKEN`` only if
    something must be downloaded. ``doi`` looks like ``"doi:10.7910/DVN/SWPENT"``. The manifest records
    each file's DOI, dataset version, Dataverse id and published MD5.
    """
    dest_dir = Path(dest_dir)
    token = os.environ.get("DATAVERSE_TOKEN")

    def api(url, *, auth=False, body=None):
        headers = dict(UA)
        if auth:
            headers["X-Dataverse-key"] = token
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers, data=body)
            ) as r:
                return json.load(r)["data"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dataverse HTTP {e.code} for {url}: {detail}") from e

    listing = (
        f"{server}/api/datasets/:persistentId/versions/{version}?persistentId={doi}"
    )
    files = {x["dataFile"]["filename"]: x["dataFile"] for x in api(listing)["files"]}
    unavailable = [n for n in filenames if n not in files]
    if unavailable:
        raise RuntimeError(
            f"Dataverse dataset {doi} version {version} is missing expected file(s): "
            f"{', '.join(unavailable)}"
        )

    def signed(access):
        """Files are guestbook-gated: an empty response returns a signed URL."""
        if not token:
            raise SystemExit(
                "set DATAVERSE_TOKEN (Dataverse account -> API Token) to download "
                f"from {doi}, or place the files in {dest_dir}"
            )
        return api(access, auth=True, body={"guestbookResponse": {}})["signedUrl"]

    for name in filenames:
        meta = files[name]
        access = f"{server}/api/access/datafile/{meta['id']}"
        upstream = {
            "doi": doi,
            "dataset_version": version,
            "file_id": meta["id"],
            "size": meta["filesize"],
            "md5": meta.get("md5") or meta.get("checksum", {}).get("value"),
        }
        download(
            lambda access=access: signed(access),
            dest_dir / name,
            version=upstream,
            source=access,
        )
