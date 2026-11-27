"""SPEI (Standardized Precipitation Evapotranspiration Index) on monthly grids.

Pure numpy: nothing here reads or writes a store, so every function works the
same on a 5x5 test array, a pilot region or one tile of the global grid.

Method (Vicente-Serrano, Begueria & Lopez-Moreno 2010; Begueria et al. 2014),
ported from MOSAIC ``pipeline/scripts/27_spei.py``:

1. Reference evapotranspiration (PET), daily, from one of two sources:

   * ``"hargreaves"`` -- Hargreaves-Samani (FAO-56, Allen et al. 1998, eq. 52)
     from CHIRTS-ERA5 Tmax/Tmin::

         ET0 = 0.0023 * Ra * (Tmean + 17.8) * sqrt(max(Tmax - Tmin, 0))   [mm/day]

     Ra is the extraterrestrial radiation in mm/day from latitude and day of
     year (FAO-56 eqs. 21, 23-25).
   * ``"agera5"`` -- AgERA5 FAO-56 Penman-Monteith ET0 (0.1 deg), summed to
     months on its own grid, then regridded bilinearly to the target grid.

2. Daily values summed to calendar months. A month with any missing day is
   missing for that cell; a month the inputs do not fully cover is dropped.
3. Climatic water balance D = P - PET, summed over ``window`` months (the label
   is the last month of the window).
4. For each cell and calendar month, a generalized logistic distribution (the
   three-parameter log-logistic of the SPEI papers) is fitted to the window sums
   of the calibration years by unbiased probability-weighted moments
   (L-moments; Hosking 1990, Hosking & Wallis 1997).
5. SPEI = standard normal quantile of the fitted cumulative probability,
   clipped to [-3, 3].

One deliberate change from 27_spei.py: the calibration months are chosen by
date only. The original also required a month to have data somewhere in the
region, which would make a cell's result depend on the tile it is processed in.
For cells with data the two rules give the same months.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

#: Fewer calibration values than this for a calendar month: no index.
MIN_CALIB = 20
#: SPEI values are clipped to this magnitude.
SPEI_CLIP = 3.0
#: |k| below this uses the k -> 0 limit of the generalized logistic.
_K_SMALL = 1e-6

PET_METHODS = ("hargreaves", "agera5")


# --------------------------------------------------------------------- PET


def ra_mm_day(doy, lat_deg) -> np.ndarray:
    """Extraterrestrial radiation as evaporation equivalent, mm/day.

    FAO-56 eqs. 21, 23, 24, 25; 0.408 converts MJ m-2 day-1 to mm/day.

    Parameters
    ----------
    doy
        Day of year, shape (T,).
    lat_deg
        Latitude in degrees, shape (Y,).

    Returns
    -------
    ndarray, shape (T, Y)
    """
    j = np.asarray(doy, "f8")[:, None]
    phi = np.deg2rad(np.asarray(lat_deg, "f8"))[None, :]
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    delta = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    ra = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    )
    return 0.408 * ra


def hargreaves_et0(tmax, tmin, ra) -> np.ndarray:
    """Daily Hargreaves-Samani ET0 in mm/day (FAO-56 eq. 52).

    ``tmax`` and ``tmin`` in degC, ``ra`` in mm/day, all broadcastable to one
    shape. A day with Tmin > Tmax gets a zero range, so ET0 = 0 for that day;
    negative results are set to 0. Missing inputs give a missing ET0.
    """
    tmax = np.asarray(tmax, "f8")
    tmin = np.asarray(tmin, "f8")
    rng = np.clip(tmax - tmin, 0, None)
    et0 = 0.0023 * ra * ((tmax + tmin) / 2 + 17.8) * np.sqrt(rng)
    return np.clip(et0, 0, None)


# ------------------------------------------------------- daily -> monthly


def month_index(dates) -> np.ndarray:
    """``datetime64`` dates -> months as ``datetime64[M]``."""
    return np.asarray(dates, "datetime64[D]").astype("datetime64[M]")


def days_in_month(months) -> np.ndarray:
    """Number of days in each ``datetime64[M]`` month."""
    m = np.asarray(months, "datetime64[M]")
    nxt = (m + np.timedelta64(1, "M")).astype("datetime64[D]")
    return (nxt - m.astype("datetime64[D]")).astype(int)


def complete_months(dates) -> np.ndarray:
    """Months whose every calendar day is in ``dates``, in order.

    ``dates`` must hold each day at most once.
    """
    d = np.unique(np.asarray(dates, "datetime64[D]"))
    months, counts = np.unique(month_index(d), return_counts=True)
    return months[counts == days_in_month(months)]


class MonthlyAccumulator:
    """Daily blocks in, calendar-month sums out, without holding the daily data.

    Feed consecutive or scattered blocks of days with :meth:`add`; read the
    result with :meth:`result`. A cell that is missing on any day of a month is
    missing for that month (the ``skipna=False`` rule of 27_spei.py), and a month
    that did not receive all of its days is missing everywhere.

    Parameters
    ----------
    months
        The ``datetime64[M]`` months to accumulate. Days outside them are ignored.
    shape
        Spatial shape of one day, e.g. (lat, lon).
    """

    def __init__(self, months, shape: tuple[int, ...]):
        self.months = np.asarray(months, "datetime64[M]")
        self._pos = {m: i for i, m in enumerate(self.months)}
        self.sums = np.zeros((len(self.months), *shape), "f8")
        self.days = np.zeros(len(self.months), int)

    def add(self, block, dates) -> None:
        """Add a block of daily values, shape (T, *shape), dated by ``dates`` (T,)."""
        block = np.asarray(block)
        mon = month_index(dates)
        for m in np.unique(mon):
            i = self._pos.get(m)
            if i is None:
                continue
            sel = mon == m
            # NaN + anything stays NaN, so one missing day blanks the month.
            self.sums[i] += block[sel].sum(axis=0, dtype="f8")
            self.days[i] += int(sel.sum())

    def result(self) -> np.ndarray:
        """Monthly sums, shape (M, *shape); incomplete months are all NaN."""
        out = self.sums.copy()
        full = self.days == days_in_month(self.months)
        if (self.days > days_in_month(self.months)).any():
            raise ValueError("a month received more days than it has (duplicate dates)")
        out[~full] = np.nan
        return out


def monthly_sums(daily, dates, months=None) -> tuple[np.ndarray, np.ndarray]:
    """One-shot version of :class:`MonthlyAccumulator`.

    Returns ``(months, sums)``. By default only the complete months of
    ``dates`` are returned.
    """
    if months is None:
        months = complete_months(dates)
    acc = MonthlyAccumulator(months, np.asarray(daily).shape[1:])
    acc.add(daily, dates)
    return acc.months, acc.result()


def common_months(*month_sets) -> np.ndarray:
    """Months present in every set, checked to be one unbroken run.

    SPEI sums consecutive months, so a gap inside the common period would make
    a window silently span more calendar time than its label says.
    """
    common = month_sets[0]
    for s in month_sets[1:]:
        common = np.intersect1d(common, s)
    common = np.asarray(common, "datetime64[M]")
    if len(common) and len(common) != (common[-1] - common[0]).astype(int) + 1:
        raise ValueError(
            f"the common months {common[0]}..{common[-1]} are not one unbroken run"
        )
    return common


# ---------------------------------------------------------- regridding


@dataclass(frozen=True)
class AxisWeights:
    """Bilinear weights along one axis: target i uses source lo[i] and lo[i]+1."""

    lo: np.ndarray
    hi: np.ndarray
    w_hi: np.ndarray  # weight of hi; weight of lo is 1 - w_hi


def axis_weights(src, dst, periodic: float | None = None) -> AxisWeights:
    """Linear-interpolation weights from a regular source axis to target points.

    ``src`` must be evenly spaced, ascending or descending. With ``periodic``
    (e.g. 360 for longitude) targets beyond the last source point wrap round
    to the first; otherwise they are clamped to the edge.
    """
    src = np.asarray(src, "f8")
    dst = np.asarray(dst, "f8")
    step = src[1] - src[0]
    pos = (dst - src[0]) / step  # fractional source index
    n = len(src)
    if periodic is not None:
        lo = np.floor(pos).astype(int)
        w_hi = pos - lo
        return AxisWeights(lo % n, (lo + 1) % n, w_hi)
    pos = np.clip(pos, 0, n - 1)
    lo = np.minimum(np.floor(pos).astype(int), n - 2)
    return AxisWeights(lo, lo + 1, pos - lo)


def regrid_bilinear(src, ylat: AxisWeights, xlon: AxisWeights) -> np.ndarray:
    """Bilinear regrid of ``src`` (..., Y, X) to the target axes.

    Missing source points are left out and the remaining weights renormalised,
    so a coastal target cell takes the value of its valid neighbours. A target
    cell with no valid neighbour is NaN.

    The weights do not depend on the data, so regridding monthly sums gives the
    same result as summing regridded days (apart from the missing-value rule).
    """
    src = np.asarray(src, "f8")
    out = np.zeros(src.shape[:-2] + (len(ylat.lo), len(xlon.lo)))
    wsum = np.zeros_like(out)
    wy = (1 - ylat.w_hi, ylat.w_hi)
    wx = (1 - xlon.w_hi, xlon.w_hi)
    for iy, ys in enumerate((ylat.lo, ylat.hi)):
        rows = src[..., ys, :]
        for ix, xs in enumerate((xlon.lo, xlon.hi)):
            v = rows[..., xs]
            w = wy[iy][:, None] * wx[ix][None, :]
            ok = np.isfinite(v) & (w > 0)
            out += np.where(ok, v, 0.0) * w
            wsum += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, out / wsum, np.nan)


# ------------------------------------------------------ accumulation


def rolling_sum(d, window: int) -> np.ndarray:
    """Sum of ``window`` consecutive months ending at each month.

    ``d`` has months on axis 0. The first ``window - 1`` months have no full
    window and are NaN; any NaN inside a window gives NaN.
    """
    d = np.asarray(d, "f8")
    tot = np.full_like(d, np.nan)
    for i in range(window - 1, d.shape[0]):
        tot[i] = d[i - window + 1 : i + 1].sum(0)
    return tot


def window_bounds(months, window: int) -> np.ndarray:
    """CF time bounds of each window, as ``datetime64[D]`` pairs (M, 2).

    The window ending in month m covers the first day of month m - window + 1
    up to (not including) the first day of month m + 1.
    """
    m = np.asarray(months, "datetime64[M]")
    one = np.timedelta64(1, "M")
    start = (m - (window - 1) * one).astype("datetime64[D]")
    end = (m + one).astype("datetime64[D]")
    return np.stack([start, end], axis=-1)


# ------------------------------------------------- generalized logistic


def glo_fit(x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generalized logistic parameters (xi, alpha, k) from a sample.

    ``x`` has the sample on axis 0, shape (n, ...). Parameters come from
    unbiased probability-weighted moments -> L-moments (Hosking 1990; the
    parameterisation of Hosking & Wallis 1997). A cell with any missing value,
    a non-positive L-scale or |k| >= 1 gets NaN for all three.
    """
    x = np.asarray(x, "f8")
    n = x.shape[0]
    xs = np.sort(x, axis=0)  # NaN sorts last; such cells are masked below
    j = np.arange(n, dtype="f8").reshape((n,) + (1,) * (x.ndim - 1))  # j - 1
    b0 = xs.mean(0)
    b1 = (xs * j / (n - 1)).sum(0) / n
    b2 = (xs * j * (j - 1) / ((n - 1) * (n - 2))).sum(0) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    with np.errstate(all="ignore"):
        k = -l3 / l2
        small = np.abs(k) < _K_SMALL
        kk = np.where(small, 1.0, k)
        alpha = np.where(small, l2, l2 * np.sin(kk * np.pi) / (kk * np.pi))
        xi = np.where(small, l1, l1 - alpha * (1 / kk - np.pi / np.sin(kk * np.pi)))
    bad = ~np.isfinite(x).all(0) | ~(l2 > 0) | ~(np.abs(k) < 1)
    xi, alpha, k = (np.array(a, "f8", ndmin=1) for a in (xi, alpha, k))
    for a in (xi, alpha, k):
        a[np.broadcast_to(bad, a.shape)] = np.nan
    if x.ndim == 1:
        return xi[0], alpha[0], k[0]
    return xi, alpha, k


def glo_cdf(x, xi, alpha, k) -> np.ndarray:
    """F(x) of the generalized logistic distribution.

    Values beyond the bounded tail get 0 or 1 (k > 0: upper bound; k < 0: lower
    bound). Missing values or parameters give NaN.
    """
    x = np.asarray(x, "f8")
    with np.errstate(all="ignore"):
        z = (x - xi) / alpha
        small = np.abs(k) < _K_SMALL
        y = 1 - np.where(small, 0.0, k) * z
        kk = np.where(small, 1.0, k)
        f = 1 / (1 + np.exp(np.log(np.where(y > 0, y, np.nan)) / kk))
        f = np.where(y > 0, f, np.where(k > 0, 1.0, 0.0))
        f = np.where(small, 1 / (1 + np.exp(-z)), f)
    return np.where(np.isfinite(x) & np.isfinite(xi + alpha + k), f, np.nan)


def glo_quantile(f, xi, alpha, k) -> np.ndarray:
    """Inverse of :func:`glo_cdf` (Hosking & Wallis 1997, A.7). Used by tests."""
    f = np.asarray(f, "f8")
    y = np.log((1 - f) / f)
    with np.errstate(all="ignore"):
        small = np.abs(k) < _K_SMALL
        kk = np.where(small, 1.0, k)
        q = np.where(small, xi - alpha * y, xi + alpha * (1 - np.exp(kk * y)) / kk)
    return q


# ------------------------------------------------------------- SPEI


def calibration_mask(months, period: tuple[int, int]) -> np.ndarray:
    """True for months whose year lies in ``period`` (inclusive), by date only."""
    years = (
        np.asarray(months, "datetime64[M]").astype("datetime64[Y]").astype(int) + 1970
    )
    return (years >= period[0]) & (years <= period[1])


def calendar_month(months) -> np.ndarray:
    """1..12 for each ``datetime64[M]`` month."""
    return np.asarray(months, "datetime64[M]").astype(int) % 12 + 1


def fit_params(
    tot, months, period: tuple[int, int], min_calib: int = MIN_CALIB
) -> np.ndarray:
    """Fitted (xi, alpha, k) per calendar month and cell.

    Parameters
    ----------
    tot
        Window sums of the water balance, shape (M, ...), months on axis 0.
    months
        ``datetime64[M]`` label of each row of ``tot``.
    period
        Calibration years, inclusive, e.g. (1991, 2020).
    min_calib
        A calendar month with fewer calibration rows is left NaN.

    Returns
    -------
    ndarray, shape (12, 3, ...)
        Index 0 is January; the second axis is (xi, alpha, k). Stored as-is, it
        lets new months be transformed later without refitting.
    """
    tot = np.asarray(tot, "f8")
    cal = calibration_mask(months, period)
    cm = calendar_month(months)
    params = np.full((12, 3) + tot.shape[1:], np.nan)
    for m in range(1, 13):
        sel = (cm == m) & cal
        if sel.sum() < min_calib:
            continue
        params[m - 1] = np.stack(glo_fit(tot[sel]))
    return params


def spei_from_params(tot, months, params) -> np.ndarray:
    """SPEI of window sums ``tot`` (M, ...) with fitted ``params`` (12, 3, ...).

    Clipped to [-3, 3]; NaN where the window sum or the fit is missing.
    """
    tot = np.asarray(tot, "f8")
    p = params[calendar_month(months) - 1]  # (M, 3, ...)
    with np.errstate(all="ignore"):
        out = norm.ppf(glo_cdf(tot, p[:, 0], p[:, 1], p[:, 2]))
    out = np.clip(out, -SPEI_CLIP, SPEI_CLIP)
    out[~np.isfinite(tot)] = np.nan
    return out


def spei(
    d, months, window: int, period: tuple[int, int], min_calib: int = MIN_CALIB
) -> tuple[np.ndarray, np.ndarray]:
    """SPEI of a monthly water balance ``d`` (M, ...), all months on axis 0.

    Returns ``(values, params)``: values has the same shape as ``d``, with the
    first ``window - 1`` months NaN (27_spei.py drops them instead).
    """
    tot = rolling_sum(d, window)
    params = fit_params(tot, months, period, min_calib)
    return spei_from_params(tot, months, params), params
