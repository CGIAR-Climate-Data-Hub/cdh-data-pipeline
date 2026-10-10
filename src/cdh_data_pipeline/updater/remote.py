"""One HTTP client for any CHC-style autoindex archive.

CHIRPS and CHIRTS are both served as plain directory listings over HTTPS, so
listing, change detection, throttling and atomic download are the same job. Only
the URL layout differs, and that is the single method a mission supplies.

Two behaviours here are not optional:

* **Serial and spaced by default.** CHC bans clients that pull hard. The rate
  limiter is shared by every request through a client, so raising the worker
  count cannot raise the request rate -- concurrency and politeness are decoupled
  on purpose.
* **A throttle is obeyed, not retried.** 429/503/509 carry a wait; retrying
  sooner than asked is how a slow-down becomes a ban.

Downloads land in a local cache directory, so ``Path`` is correct here -- see the
destination note in CONVENTIONS.md.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from cdh_data_pipeline.utils import (
    AUTOINDEX_ROW,
    SIZE_TOLERANCE,
    parse_dotted_date,
    parse_listing_size,
)

from .base import RateLimiter, throttle_delay

USER_AGENT = "cdh-data-pipeline/0.1 (climate datacube builder)"


@dataclass(frozen=True)
class RemoteFile:
    """One dated raster of one variable on a remote archive.

    ``size`` comes from the directory listing and is therefore approximate --
    autoindex rounds to one decimal of a binary unit. Exact verification happens
    at download time against the response's ``Content-Length``.
    """

    variable: str
    day: date
    url: str
    size: int | None
    modified: str

    @property
    def signature(self) -> str:
        """Changes when the file is reprocessed upstream.

        Size plus modification time, both from the listing. "Have I seen this date
        before" is not enough: CHC republishes months with new content.
        """
        return f"{self.size}:{self.modified}"

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


class ArchiveRemote:
    """Listing and download over an Apache/nginx autoindex.

    Subclasses implement :meth:`year_url`. Everything else -- pacing, retry,
    throttle handling, listing, atomic download -- is shared.
    """

    #: Variable keys this archive serves; used to validate callers.
    variables: tuple[str, ...] = ()

    def __init__(
        self, timeout: int = 120, retries: int = 4, min_interval: float = 1.0
    ):
        self.timeout = timeout
        self.retries = retries
        #: Shared across every request, so worker count cannot outpace it.
        self.limiter = RateLimiter(min_interval)

    # ----------------------------------------------------------- to implement

    def year_url(self, variable: str, year: int) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------ http

    def _open(self, url: str):
        self.limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _retry(self, fn, what: str):
        last = None
        for attempt in range(self.retries):
            try:
                return fn()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                pause = throttle_delay(exc, attempt)
                if pause is not None:
                    # Hold every other request back too, not just this one.
                    self.limiter.back_off(pause)
                    print(
                        f"    server throttled us; waiting {pause:.0f}s", flush=True
                    )
                    time.sleep(pause)
                elif attempt < self.retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"{what} failed after {self.retries} attempts: {last}")

    # --------------------------------------------------------------- listing

    def list_year(self, variable: str, year: int) -> list[RemoteFile]:
        """Every published day of ``variable`` in ``year`` -- one request."""
        if self.variables and variable not in self.variables:
            raise KeyError(
                f"unknown variable {variable!r}; "
                f"expected one of {sorted(self.variables)}"
            )
        url = self.year_url(variable, year)

        def fetch():
            with self._open(url) as resp:
                return resp.read().decode("utf-8", errors="replace")

        try:
            html = self._retry(fetch, f"listing {url}")
        except RuntimeError:
            return []  # year directory absent or empty

        out = []
        for m in AUTOINDEX_ROW.finditer(html):
            name = m.group("name")
            try:
                day = parse_dotted_date(name)
            except ValueError:
                continue
            out.append(
                RemoteFile(
                    variable=variable,
                    day=day,
                    url=f"{url}{name}",
                    size=parse_listing_size(m.group("size")),
                    modified=m.group("date").strip(),
                )
            )
        out.sort(key=lambda r: r.day)
        return out

    def latest_day(self, variable: str, search_back: int = 2) -> date | None:
        today = date.today()
        for offset in range(search_back + 1):
            rows = self.list_year(variable, today.year - offset)
            if rows:
                return rows[-1].day
        return None

    def complete_months(self, variable: str, year: int) -> set[tuple[int, int]]:
        """Months with every calendar day published.

        Only meaningful for archives that release a whole month at a time; CHIRTS
        trickles individual days and should not gate on this.
        """
        import calendar

        by_month: dict[int, set[int]] = {}
        for r in self.list_year(variable, year):
            by_month.setdefault(r.day.month, set()).add(r.day.day)
        return {
            (year, m)
            for m, days in by_month.items()
            if len(days) == calendar.monthrange(year, m)[1]
        }

    # -------------------------------------------------------------- download

    def download(
        self, remote: RemoteFile, cache_dir: str | Path, overwrite: bool = False
    ) -> Path:
        """Fetch one raster into the local cache, writing atomically via ``.part``."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / remote.filename
        if target.exists() and not overwrite:
            if (
                remote.size is None
                or abs(target.stat().st_size - remote.size)
                <= remote.size * SIZE_TOLERANCE
            ):
                return target

        tmp = target.with_suffix(target.suffix + ".part")

        def fetch():
            written = 0
            with self._open(remote.url) as resp, open(tmp, "wb") as fh:
                declared = resp.headers.get("Content-Length")
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
            return written, (int(declared) if declared else None)

        got, declared = self._retry(fetch, f"download {remote.url}")
        # Content-Length is exact, unlike the listing: a short read is a truncated
        # transfer and must never pass as a valid raster.
        if declared is not None and got != declared:
            tmp.unlink(missing_ok=True)
            raise OSError(
                f"{remote.filename}: truncated, got {got} of {declared} bytes"
            )
        tmp.replace(target)
        return target

    def download_many(
        self,
        remotes: Sequence[RemoteFile],
        cache_dir: str | Path,
        workers: int = 1,
        progress: bool = True,
    ) -> list[Path | None]:
        """Fetch rasters, serially by default.

        Concurrency is opt-in; the shared rate limiter caps the request rate
        regardless of it. The result is positional: a failed download leaves
        ``None`` at that index so the caller can still line paths up with the days
        it asked for.
        """
        paths: list[Path | None] = [None] * len(remotes)

        def one(item):
            i, r = item
            try:
                paths[i] = self.download(r, cache_dir)
            except Exception as exc:  # noqa: BLE001  reported by the caller
                print(f"    {r.filename}: {type(exc).__name__}: {exc}", flush=True)

        t0 = time.time()
        with ThreadPoolExecutor(max(1, workers)) as ex:
            for k, _ in enumerate(ex.map(one, enumerate(remotes)), 1):
                if progress and (k % 10 == 0 or k == len(remotes)):
                    print(
                        f"    downloaded {k}/{len(remotes)} [{time.time() - t0:.0f}s]",
                        flush=True,
                    )
        return paths


class ChirpsArchive(ArchiveRemote):
    """CHIRPS v3 daily, ``products/CHIRPS/v3.0/daily/<kind>/<flavour>/<year>/``.

    ``sat`` and ``rnl`` are different products, not different coverage of one: the
    same pentads disaggregated with IMERG and with ERA5. ``sat`` starts in 1998
    because IMERG does; ``rnl`` runs from 1981.
    """

    BASE = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily"
    variables = ("precip",)

    def __init__(self, flavour: str = "rnl", kind: str = "final", **kw):
        if flavour not in ("sat", "rnl"):
            raise ValueError(f"flavour must be 'sat' or 'rnl', got {flavour!r}")
        super().__init__(**kw)
        self.flavour = flavour
        self.kind = kind

    def year_url(self, variable: str, year: int) -> str:
        return f"{self.BASE}/{self.kind}/{self.flavour}/{year}/"


class ChirtsArchive(ArchiveRemote):
    """CHIRTS-ERA5 daily, ``experimental/CHIRTS-ERA5/<var>/tifs/daily/<year>/``.

    Note ``/experimental/``, not ``/products/``: the ``CHIRTSdaily`` published
    alongside is the station-blended v1.0, frozen at 1983-2016. This one is the
    ERA5 reconstruction, 1959 to present.
    """

    BASE = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5"
    variables = ("tmax", "tmin")
    FOLDERS = {"tmax": "tmax", "tmin": "tmin"}

    def year_url(self, variable: str, year: int) -> str:
        return f"{self.BASE}/{self.FOLDERS[variable]}/tifs/daily/{year}/"
