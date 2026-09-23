"""Download source files into a local input cache.

``download`` covers plain URLs. ``download_dataverse`` handles the three quirks
that trip up plain urllib against Dataverse: a WAF that 403s the default Python
user agent, guestbook-gated files (POST a guestbook response to get a signed URL),
and streaming the bytes from that signed URL.
"""

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from cdh_data_pipeline.recipe import log

HARVARD = "https://dataverse.harvard.edu"
UA = {"User-Agent": "cdh-data-pipeline"}


def download(url, dest):
    """Download ``url`` to ``dest`` unless it already exists. Returns ``dest``.

    Streams to ``dest.part`` and renames on completion so an interrupted download
    is never mistaken for a complete file on the next run.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log.info("downloading %s", dest.name)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(dest)
    log.info("downloaded %s (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_dataverse(doi, filenames, dest_dir, *, version=":latest", server=HARVARD):
    """Download ``filenames`` from a Dataverse dataset into ``dest_dir``.

    Files already present are skipped, so ``DATAVERSE_TOKEN`` (from your account's
    API Token page) is only needed when something must actually be fetched. ``doi``
    is the dataset persistent id, e.g. ``"doi:10.7910/DVN/SWPENT"``; ``version`` is
    a Dataverse version such as ``"6.0"`` or ``":latest"``.
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
        if body is not None:  # a JSON body makes it a POST
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
        # Guestbook-gated: POST an (empty) guestbook response -- name, email,
        # institution default to the token's account -- to get a signed, tokened
        # URL that needs no auth header.
        access = f"{server}/api/access/datafile/{ids[name]}"
        signed = api(access, auth=True, body={"guestbookResponse": {}})["signedUrl"]
        download(signed, dest_dir / name)
