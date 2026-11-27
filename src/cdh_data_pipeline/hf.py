"""Publish a local dataset prefix to a Hugging Face dataset repository.

A monthly update rewrites a few hundred chunks of a store holding tens of
thousands, so a publish sends only the files whose *content* differs from the
repo, then confirms the repo matches.

How "differs" is decided
------------------------
The repo listing already carries a content hash for every file: the git blob
SHA-1 for files kept in git, the SHA-256 for LFS/Xet files. The same hashes are
computed locally and compared. Sizes alone are not enough: an attribute edit that
turns ``2026-08-31`` into ``2026-10-31`` leaves ``zarr.json`` the same size. No
snapshot from a previous run is needed, so any machine holding the store can
publish it, and a half-finished earlier push is simply finished.

Local hashes are cached in ``<prefix>/.hf-hashes.json`` keyed by size and
modification time, so only files touched since the last publish are re-read.

Order of writes
---------------
Chunk files go first, in batches; every ``zarr.json``, the deletions and the
sidecars go in the final commit. A reader of the repo therefore never sees
metadata that points past the chunks already uploaded, and an interrupted publish
is resumed by running it again.

Guards
------
* the local store must have ``build_complete`` set;
* no array may be shorter locally than in the repo (an old copy published over a
  newer one), unless ``force``;
* deletions above ``max_delete_frac`` of the repo's store files stop the run,
  unless ``force``.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from cdh_data_pipeline.recipe import log

HASH_CACHE = ".hf-hashes.json"
#: Files in one commit. Large enough that a monthly update is one or two
#: commits, small enough that a failed commit costs little to redo.
BATCH = 1000


def _hf():
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "publishing needs huggingface_hub: pip install 'huggingface_hub[hf_xet]'"
        ) from exc
    return huggingface_hub


@dataclass(frozen=True)
class RemoteFile:
    size: int
    blob_id: str
    sha256: str | None  # set for LFS/Xet files


def digests(path: Path) -> tuple[str, str]:
    """``(git blob SHA-1, SHA-256)`` of a file, in one read."""
    sha1 = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            sha1.update(block)
            sha256.update(block)
    return sha1.hexdigest(), sha256.hexdigest()


def same_content(local: tuple[str, str], remote: RemoteFile) -> bool:
    sha1, sha256 = local
    return sha256 == remote.sha256 if remote.sha256 else sha1 == remote.blob_id


@dataclass
class PublishPlan:
    upload: list[str] = field(default_factory=list)  # repo paths
    delete: list[str] = field(default_factory=list)
    unchanged: int = 0
    bytes: int = 0

    @property
    def empty(self) -> bool:
        return not (self.upload or self.delete)

    def summary(self) -> str:
        if self.empty:
            return f"in sync ({self.unchanged:,} file(s) unchanged)"
        return (
            f"{len(self.upload):,} to upload ({self.bytes / 1e6:,.1f} MB), "
            f"{len(self.delete):,} to delete, {self.unchanged:,} unchanged"
        )


class HashCache:
    """Local digests keyed by repo path, reused while size and mtime hold."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.entries = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            self.entries = {}

    def get(self, key: str, path: Path) -> tuple[str, str]:
        st = path.stat()
        hit = self.entries.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2], hit[3]
        sha1, sha256 = digests(path)
        self.entries[key] = [st.st_size, st.st_mtime_ns, sha1, sha256]
        return sha1, sha256

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries), encoding="utf-8")
        tmp.replace(self.path)


def remote_files(api, repo_id: str) -> dict[str, RemoteFile]:
    """Every file in the repo -> size and content hash; {} if the repo is absent."""
    hf = _hf()
    try:
        items = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        out = {}
        for item in items:
            size = getattr(item, "size", None)
            if size is None:  # a folder
                continue
            lfs = getattr(item, "lfs", None)
            out[item.path] = RemoteFile(size, item.blob_id, lfs.sha256 if lfs else None)
        return out
    except hf.errors.RepositoryNotFoundError:
        return {}


def local_files(prefix: Path, store_name: str, sidecars) -> dict[str, Path]:
    """Repo path -> local file, for the store and whichever sidecars exist."""
    store = prefix / store_name
    out = {
        f"{store_name}/{f.relative_to(store).as_posix()}": f
        for f in store.rglob("*")
        if f.is_file()
    }
    for name in sidecars:
        if (prefix / name).is_file():
            out[name] = prefix / name
    return out


def plan_publish(
    local: dict[str, Path],
    remote: dict[str, RemoteFile],
    store_name: str,
    cache: HashCache,
    workers: int = 8,
) -> PublishPlan:
    """What to upload and delete so the repo matches ``local``.

    Deletions are limited to the store: a chunk that became all-nodata is removed
    from disk, and a stale copy in the repo would disagree with it. Sidecars and
    other repo files (the card, catalog records) are never deleted.
    """
    plan = PublishPlan()
    to_hash = []
    for key, path in sorted(local.items()):
        r = remote.get(key)
        if r is None or r.size != path.stat().st_size:
            plan.upload.append(key)
        else:
            to_hash.append((key, path, r))
    with ThreadPoolExecutor(workers) as pool:
        same = list(
            pool.map(lambda t: same_content(cache.get(t[0], t[1]), t[2]), to_hash)
        )
    for (key, _, _), ok in zip(to_hash, same):
        if ok:
            plan.unchanged += 1
        else:
            plan.upload.append(key)
    plan.upload.sort()
    plan.bytes = sum(local[k].stat().st_size for k in plan.upload)
    plan.delete = sorted(
        k for k in remote if k.startswith(f"{store_name}/") and k not in local
    )
    return plan


def _array_shapes(zarr_json: dict) -> dict[str, list[int]]:
    meta = (zarr_json.get("consolidated_metadata") or {}).get("metadata") or {}
    return {k: v["shape"] for k, v in meta.items() if v.get("node_type") == "array"}


def check_not_older(api, repo_id: str, store: Path, store_name: str) -> None:
    """Refuse to publish a store with any array shorter than the repo's copy."""
    path = _hf().hf_hub_download(
        repo_id, f"{store_name}/zarr.json", repo_type="dataset"
    )
    remote = _array_shapes(json.loads(Path(path).read_text(encoding="utf-8")))
    local = _array_shapes(json.loads((store / "zarr.json").read_text(encoding="utf-8")))
    shorter = [
        f"{name} {local[name]} < {shape}"
        for name, shape in remote.items()
        if name in local and any(a < b for a, b in zip(local[name], shape))
    ]
    if shorter:
        raise ValueError(
            "the repo holds a larger store than this one, so publishing would "
            "roll it back: " + "; ".join(shorter) + ". Update this copy first, "
            "or pass force=True if the shrink is intended."
        )


def _commit(api, repo_id, adds, deletes, local, message):
    hf = _hf()
    ops = [
        hf.CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(local[k]))
        for k in adds
    ]
    ops += [hf.CommitOperationDelete(path_in_repo=k) for k in deletes]
    api.create_commit(
        repo_id=repo_id, repo_type="dataset", operations=ops, commit_message=message
    )


def publish(
    prefix: str | os.PathLike,
    repo_id: str,
    store_name: str,
    sidecars=("README.md", "sources.json"),
    *,
    message: str | None = None,
    private: bool = False,
    dry_run: bool = False,
    force: bool = False,
    max_delete_frac: float = 0.01,
) -> dict:
    """Make ``repo_id`` hold ``<prefix>/<store_name>`` and the sidecars, exactly.

    The store lands at ``<store_name>/`` in the repo and each sidecar at the repo
    root. The first publish uploads the whole folder (``upload_folder`` splits it
    into resumable commits); later ones send only what changed.
    """
    hf = _hf()
    prefix = Path(prefix)
    store = prefix / store_name
    if not store.is_dir():
        raise FileNotFoundError(f"{store} not found; publish needs a local store")
    attrs = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    if not attrs.get("attributes", {}).get("build_complete"):
        raise RuntimeError(
            f"{store} is not finished (no build_complete); not publishing"
        )

    api = hf.HfApi()
    log.info("publishing %s to %s as %s", store, repo_id, api.whoami()["name"])
    message = message or f"Update {store_name}"
    remote = remote_files(api, repo_id)
    has_store = any(k.startswith(f"{store_name}/") for k in remote)

    if has_store and not force:
        check_not_older(api, repo_id, store, store_name)

    if not has_store:
        log.info("%s is not in the repo yet; uploading the whole store", store_name)
        if dry_run:
            return {"dry_run": True, "initial": True}
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(store),
            path_in_repo=store_name,
            commit_message=message,
        )
        remote = remote_files(api, repo_id)

    cache = HashCache(prefix / HASH_CACHE)
    local = local_files(prefix, store_name, sidecars)
    plan = plan_publish(local, remote, store_name, cache)
    cache.save()
    log.info("plan: %s", plan.summary())

    n_store = sum(k.startswith(f"{store_name}/") for k in remote)
    if len(plan.delete) > max_delete_frac * max(n_store, 1) and not force:
        raise RuntimeError(
            f"{len(plan.delete):,} of {n_store:,} store files would be deleted from "
            f"the repo, e.g. {plan.delete[:3]}; pass force=True if that is intended"
        )
    counts = {
        "upload": len(plan.upload),
        "delete": len(plan.delete),
        "unchanged": plan.unchanged,
        "bytes": plan.bytes,
    }
    if dry_run or plan.empty:
        return {"dry_run": dry_run, **counts}

    # Chunks first, metadata and sidecars last; see the module docstring.
    last = [k for k in plan.upload if k.endswith("zarr.json") or "/" not in k]
    data = [k for k in plan.upload if k not in set(last)]
    for i in range(0, len(data), BATCH):
        part = data[i : i + BATCH]
        log.info("  commit %d-%d of %d chunk file(s)", i + 1, i + len(part), len(data))
        _commit(api, repo_id, part, [], local, f"{message} (data {i // BATCH + 1})")
    log.info(
        "  commit metadata: %d file(s), %d deletion(s)", len(last), len(plan.delete)
    )
    _commit(api, repo_id, last, plan.delete, local, message)

    after = plan_publish(local, remote_files(api, repo_id), store_name, cache)
    cache.save()
    if not after.empty:
        raise RuntimeError(
            f"the repo still differs after publishing: {after.summary()}"
        )
    log.info("confirmed: repo matches the local store")
    return {
        "dry_run": False,
        **counts,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }
