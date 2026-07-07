"""Download source files from a Dataverse dataset (Harvard, IFPRI, etc.).

Handles the three quirks that trip up plain urllib against Dataverse: a WAF that
403s the default Python user agent, guestbook-gated files (POST a guestbook
response to get a signed URL), and streaming the bytes from that signed URL.
"""

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

HARVARD = "https://dataverse.harvard.edu"


def download_dataverse(doi, filenames, dest_dir, *, version=":latest", server=HARVARD):
    """Download ``filenames`` from a Dataverse dataset into ``dest_dir``.

    Files already present are skipped, so ``DATAVERSE_TOKEN`` (from your account's
    API Token page) is only needed when something must actually be fetched. ``doi``
    is the dataset persistent id, e.g. ``"doi:10.7910/DVN/SWPENT"``; ``version`` is
    a Dataverse version such as ``"6.0"`` or ``":latest"``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in filenames if not (dest_dir / n).exists()]
    if not missing:
        return

    token = os.environ.get("DATAVERSE_TOKEN")
    if not token:
        raise SystemExit(
            "set DATAVERSE_TOKEN (Dataverse account -> API Token) to download "
            f"{len(missing)} file(s), or place them in {dest_dir}"
        )

    def req(url, *, auth=False, body=None):
        # The WAF 403s the default Python-urllib user agent. A JSON body makes it
        # a POST; otherwise GET.
        headers = {"User-Agent": "cdh-data-pipeline"}
        if auth:
            headers["X-Dataverse-key"] = token
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body).encode()
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=headers, data=body)
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dataverse HTTP {e.code} for {url}: {detail}") from e

    listing = (
        f"{server}/api/datasets/:persistentId/versions/{version}?persistentId={doi}"
    )
    files = json.load(req(listing))["data"]["files"]
    ids = {x["dataFile"]["filename"]: x["dataFile"]["id"] for x in files}
    for name in missing:
        print(f"  downloading {name}")
        # Guestbook-gated: POST an (empty) guestbook response -- name, email,
        # institution default to the token's account -- to get a signed, tokened
        # URL, then stream the bytes from it (that URL needs no auth header).
        access = f"{server}/api/access/datafile/{ids[name]}"
        signed = json.load(req(access, auth=True, body={"guestbookResponse": {}}))
        with req(signed["data"]["signedUrl"]) as r, open(dest_dir / name, "wb") as f:
            shutil.copyfileobj(r, f)
