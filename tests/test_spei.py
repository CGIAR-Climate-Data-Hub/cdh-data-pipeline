"""Core SPEI functions on small synthetic arrays."""

import numpy as np
import pytest
from scipy.stats import norm

from cdh_data_pipeline import spei as S

PERIOD = (1991, 2020)


def months_range(first: str, last: str) -> np.ndarray:
    return np.arange(
        np.datetime64(first, "M"), np.datetime64(last, "M") + np.timedelta64(1, "M")
    )


def days_range(first: str, last: str) -> np.ndarray:
    return np.arange(
        np.datetime64(first, "D"), np.datetime64(last, "D") + np.timedelta64(1, "D")
    )


# ----------------------------------------------------------------- PET


def test_ra_matches_fao56_example_8():
    # FAO-56 Example 8: 20 deg S, 3 September (day 246) -> Ra = 32.2 MJ m-2 d-1.
    ra = S.ra_mm_day([246], [-20.0])[0, 0]
    assert ra == pytest.approx(0.408 * 32.2, abs=0.05)


def test_ra_polar_night_is_zero():
    # 21 December at 80 N: the sun does not rise.
    assert S.ra_mm_day([355], [80.0])[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_hargreaves_by_hand_and_inverted_range():
    ra = 15.0
    got = S.hargreaves_et0(np.array([30.0, 20.0]), np.array([20.0, 25.0]), ra)
    assert got[0] == pytest.approx(0.0023 * 15 * (25 + 17.8) * np.sqrt(10))
    assert got[1] == 0.0  # Tmin > Tmax: zero range, so no ET0
    assert np.isnan(S.hargreaves_et0(np.nan, 10.0, ra))


# ------------------------------------------------------ daily -> monthly


def test_complete_months_drops_partial_ends():
    d = days_range("2020-01-15", "2020-04-30")
    assert list(S.complete_months(d).astype(str)) == ["2020-02", "2020-03", "2020-04"]


def test_monthly_sums_missing_day_blanks_only_that_cell_and_month():
    d = days_range("2020-01-01", "2020-03-31")
    x = np.ones((len(d), 2, 2))
    x[40, 0, 1] = np.nan  # 10 February, one cell
    months, m = S.monthly_sums(x, d)
    assert list(months.astype(str)) == ["2020-01", "2020-02", "2020-03"]
    assert m[0, 0, 0] == 31 and m[1, 0, 0] == 29 and m[2, 1, 1] == 31
    assert np.isnan(m[1, 0, 1])
    assert np.isfinite(m[0, 0, 1]) and np.isfinite(m[2, 0, 1])


def test_accumulator_streaming_equals_one_shot_and_flags_short_months():
    rng = np.random.default_rng(0)
    d = days_range("2019-12-01", "2020-03-31")
    x = rng.gamma(1.0, 3.0, size=(len(d), 3, 4))
    want_months, want = S.monthly_sums(x, d)
    acc = S.MonthlyAccumulator(want_months, (3, 4))
    for i in range(0, len(d), 17):  # blocks that cut through months
        acc.add(x[i : i + 17], d[i : i + 17])
    np.testing.assert_allclose(acc.result(), want)
    # A month that never received all its days is missing everywhere.
    short = S.MonthlyAccumulator(want_months, (3, 4))
    short.add(x[:-1], d[:-1])
    assert np.isnan(short.result()[-1]).all()


def test_common_months_requires_unbroken_run():
    a = months_range("1981-01", "2026-10")
    b = months_range("1981-01", "2026-11")
    assert S.common_months(a, b)[-1] == np.datetime64("2026-10")
    with pytest.raises(ValueError):
        S.common_months(np.delete(a, 5), b)


# ---------------------------------------------------------- regridding


def test_bilinear_reproduces_a_linear_field_on_the_chirps_offsets():
    # AgERA5-like nodes at multiples of 0.1, latitude ascending; CHIRPS-like
    # cell centres at .025/.075, latitude descending.
    src_lat = np.round(np.arange(-1.0, 1.01, 0.1), 6)
    src_lon = np.round(np.arange(10.0, 11.01, 0.1), 6)
    field = 2.0 * src_lat[:, None] + 3.0 * src_lon[None, :] + 1.0
    dst_lat = np.round(np.arange(0.975, -0.98, -0.05), 6)
    dst_lon = np.round(np.arange(10.025, 10.98, 0.05), 6)
    got = S.regrid_bilinear(
        field, S.axis_weights(src_lat, dst_lat), S.axis_weights(src_lon, dst_lon)
    )
    want = 2.0 * dst_lat[:, None] + 3.0 * dst_lon[None, :] + 1.0
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_bilinear_renormalises_round_missing_points_and_wraps_longitude():
    src_lat = np.array([0.0, 0.1])
    src_lon = np.round(np.arange(-180.0, 180.0, 0.1), 6)
    field = np.ones((2, len(src_lon)))
    field[:, -1] = 3.0  # 179.9
    field[0, 0] = np.nan  # -180.0, lat 0.0: missing (e.g. sea)
    ylat = S.axis_weights(src_lat, np.array([0.025]))
    # 179.975 lies between 179.9 and -180.0 (= 180.0): the axis wraps.
    xlon = S.axis_weights(src_lon, np.array([179.975]), periodic=360.0)
    assert (xlon.lo[0], xlon.hi[0]) == (len(src_lon) - 1, 0)
    got = S.regrid_bilinear(field, ylat, xlon)[0, 0]
    # weights: (lat0,179.9)=.75*.25, (lat0,-180)=missing, (lat1,179.9)=.25*.25, (lat1,-180)=.25*.75
    w = np.array([0.75 * 0.25, 0.25 * 0.25, 0.25 * 0.75])
    assert got == pytest.approx((w * [3.0, 3.0, 1.0]).sum() / w.sum())
    all_missing = np.full_like(field, np.nan)
    assert np.isnan(S.regrid_bilinear(all_missing, ylat, xlon)[0, 0])


# ------------------------------------------------------ accumulation


def test_rolling_sum_edges_and_nan():
    d = np.arange(1.0, 7.0)
    np.testing.assert_allclose(S.rolling_sum(d, 3), [np.nan, np.nan, 6, 9, 12, 15])
    d[3] = np.nan
    got = S.rolling_sum(d, 3)
    assert np.isnan(got[3:6]).all() and got[2] == 6
    np.testing.assert_allclose(S.rolling_sum(np.arange(1.0, 4.0), 1), [1, 2, 3])


def test_window_bounds():
    b = S.window_bounds(np.array(["2020-03", "2021-01"], "datetime64[M]"), 3)
    assert b.astype(str).tolist() == [
        ["2020-01-01", "2020-04-01"],
        ["2020-11-01", "2021-02-01"],
    ]


# ------------------------------------------------- generalized logistic


@pytest.mark.parametrize("k", [-0.3, -0.05, 0.0, 0.2])
def test_glo_fit_recovers_parameters(k):
    rng = np.random.default_rng(42)
    xi, alpha = 50.0, 20.0
    x = S.glo_quantile(rng.uniform(size=200_000), xi, alpha, k)
    fxi, falpha, fk = S.glo_fit(x)
    assert fxi == pytest.approx(xi, abs=0.5)
    assert falpha == pytest.approx(alpha, rel=0.02)
    assert fk == pytest.approx(k, abs=0.01)


@pytest.mark.parametrize("k", [-0.4, 0.0, 0.3])
def test_glo_cdf_inverts_quantile(k):
    f = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(
        S.glo_cdf(S.glo_quantile(f, 10.0, 3.0, k), 10.0, 3.0, k), f, atol=1e-12
    )


def test_glo_cdf_bounded_tails():
    # k > 0 has an upper bound at xi + alpha/k; k < 0 a lower bound.
    assert S.glo_cdf(100.0, 0.0, 1.0, 0.5) == 1.0
    assert S.glo_cdf(-100.0, 0.0, 1.0, -0.5) == 0.0
    assert np.isnan(S.glo_cdf(1.0, np.nan, 1.0, 0.1))


def test_glo_fit_masks_bad_cells():
    x = np.random.default_rng(1).normal(size=(30, 3))
    x[4, 0] = np.nan  # missing value
    x[:, 1] = 7.0  # constant: L-scale 0
    xi, alpha, k = S.glo_fit(x)
    assert np.isnan([xi[0], alpha[0], k[0], xi[1], alpha[1], k[1]]).all()
    assert np.isfinite([xi[2], alpha[2], k[2]]).all()


# ------------------------------------------------------------- SPEI


def reference_spei(d, labels, window, normal, min_calib=20):
    """The ``spei`` function of MOSAIC 27_spei.py, copied verbatim (logging removed)."""
    tot = np.full_like(d, np.nan)
    for i in range(window - 1, d.shape[0]):
        tot[i] = d[i - window + 1 : i + 1].sum(0)
    months = np.array([int(lb[5:7]) for lb in labels])
    years = np.array([int(lb[:4]) for lb in labels])
    calib = (
        (years >= normal[0]) & (years <= normal[1]) & np.isfinite(tot).any(axis=(1, 2))
    )
    out = np.full_like(tot, np.nan)
    for m in range(1, 13):
        sel = months == m
        cal = sel & calib
        if cal.sum() < min_calib:
            continue
        xi, alpha, k = S.glo_fit(tot[cal])
        with np.errstate(all="ignore"):
            out[sel] = norm.ppf(S.glo_cdf(tot[sel], xi[None], alpha[None], k[None]))
    out = np.clip(out, -3, 3)
    out[~np.isfinite(tot)] = np.nan
    return out[window - 1 :], labels[window - 1 :]


def synthetic_balance(shape=(4, 5), first="1981-01", last="2026-10", seed=3):
    months = months_range(first, last)
    rng = np.random.default_rng(seed)
    season = 60 * np.sin(2 * np.pi * (S.calendar_month(months) - 4) / 12)
    p = (
        rng.gamma(2.0, 40.0, size=(len(months), *shape))
        + np.clip(season, 0, None)[:, None, None]
    )
    pet = 90 + 30 * np.cos(2 * np.pi * (S.calendar_month(months) - 6) / 12)
    return months, p - pet[:, None, None]


@pytest.mark.parametrize("window", [3, 6, 12])
def test_spei_matches_27_spei_reference(window):
    months, d = synthetic_balance()
    d[100, 1, 2] = np.nan  # one missing month in one cell (1989-05, before calibration)
    labels = list(months.astype(str))
    want, _ = reference_spei(d, labels, window, PERIOD)
    got, params = S.spei(d, months, window, PERIOD)
    assert np.isnan(got[: window - 1]).all()
    np.testing.assert_array_equal(np.isnan(got[window - 1 :]), np.isnan(want))
    np.testing.assert_allclose(got[window - 1 :], want, atol=1e-12, equal_nan=True)
    assert params.shape == (12, 3, 4, 5)


def test_spei_is_standard_normal_over_calibration_and_clipped():
    months, d = synthetic_balance(shape=(3, 3))
    got, _ = S.spei(d, months, 3, PERIOD)
    cal = S.calibration_mask(months, PERIOD)
    v = got[cal]
    assert abs(np.nanmean(v)) < 0.1
    assert np.nanstd(v) == pytest.approx(1.0, abs=0.1)
    assert np.nanmax(np.abs(got)) <= 3.0


def test_spei_from_stored_params_equals_full_run():
    # Incremental updates transform new months with stored parameters.
    months, d = synthetic_balance()
    full, params = S.spei(d, months, 3, PERIOD)
    tot = S.rolling_sum(d, 3)
    np.testing.assert_allclose(
        S.spei_from_params(tot[-6:], months[-6:], params), full[-6:]
    )


def test_spei_cell_does_not_depend_on_tile():
    months, d = synthetic_balance(shape=(4, 6))
    d[:, :, :3] = np.nan  # left half of the tile is sea
    full, _ = S.spei(d, months, 3, PERIOD)
    right, _ = S.spei(d[:, :, 3:], months, 3, PERIOD)
    np.testing.assert_allclose(full[:, :, 3:], right)
    assert np.isnan(full[:, :, :3]).all()


def test_spei_too_few_calibration_years_leaves_month_empty():
    months, d = synthetic_balance(shape=(2, 2), first="2000-01", last="2026-10")
    got, params = S.spei(d, months, 3, (2001, 2020))  # 20 years: enough
    assert np.isfinite(params).all()
    got, params = S.spei(d, months, 3, (2002, 2020))  # 19 years: no fit
    assert np.isnan(params).all() and np.isnan(got).all()
