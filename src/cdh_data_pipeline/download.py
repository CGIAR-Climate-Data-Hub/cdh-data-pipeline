"""Download source files into a local input cache."""

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from cdh_data_pipeline.recipe import log

HARVARD = "https://dataverse.harvard.edu"
UA = {"User-Agent": "cdh-data-pipeline"}  # Dataverse blocks urllib's default


def download(url, dest):
    """Download ``url`` to ``dest`` unless it exists. Returns ``dest``."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")  # so a partial file never looks done
    log.info("downloading %s", dest.name)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(dest)
    log.info("downloaded %s (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_dataverse(doi, filenames, dest_dir, *, version=":latest", server=HARVARD):
    """Download ``filenames`` from a Dataverse dataset into ``dest_dir``.

    Skips files already present. Needs ``DATAVERSE_TOKEN`` only if something is
    missing. ``doi`` looks like ``"doi:10.7910/DVN/SWPENT"``.
    """
    dest_dir = Path(dest_dir)
    missing = [n for n in filenames if not (dest_dir / n).exists()]
    if not missing:
        return

    token = os.environ.get("DATAVERSE_TOKEN")
    if not token:
        raise SystemExit(
            "set DATAVERSE_TOKEN (Dataverse account -> API Token) to download "
            f"{len(missing)} file(s), or place them in {dest_dir}"
        )

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
    ids = {
        x["dataFile"]["filename"]: x["dataFile"]["id"] for x in api(listing)["files"]
    }
    unavailable = [n for n in missing if n not in ids]
    if unavailable:
        raise RuntimeError(
            f"Dataverse dataset {doi} version {version} is missing expected file(s): "
            f"{', '.join(unavailable)}"
        )
    for name in missing:
        # Files are guestbook-gated: an empty response returns a signed URL.
        access = f"{server}/api/access/datafile/{ids[name]}"
        signed = api(access, auth=True, body={"guestbookResponse": {}})["signedUrl"]
        download(signed, dest_dir / name)
