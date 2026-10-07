"""check_packable guards integer packing against silent overflow."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from cdh_data_pipeline import check_packable

INT16_TENTHS = {"dtype": "int16", "scale_factor": 0.1, "_FillValue": -32768}


def dataset(values, chunks=None):
    data = np.asarray(values, dtype="float32")
    return xr.Dataset({"v": ("t", da.from_array(data, chunks) if chunks else data)})


def test_values_inside_the_packed_range_pass():
    ds = dataset([np.nan, -3276.7, 0, 0.04, 3276.7])
    xr.testing.assert_identical(check_packable(ds, {"v": INT16_TENTHS}), ds)


@pytest.mark.parametrize("value", [3276.8, 4000, -3276.8, np.inf])
def test_overflow_raises(value):
    with pytest.raises(ValueError, match="overflows"):
        check_packable(dataset([0, value]), {"v": INT16_TENTHS})


def test_dask_data_is_checked_lazily_per_chunk():
    ds = check_packable(dataset([0, 1, 4000], chunks=1), {"v": INT16_TENTHS})
    with pytest.raises(ValueError, match="overflows"):
        ds.v.compute()


def test_offset_and_unsigned_fill_shift_the_range():
    enc = {"dtype": "uint8", "scale_factor": 0.5, "add_offset": 10, "_FillValue": 255}
    check_packable(dataset([10, 137]), {"v": enc})  # 0..254 * 0.5 + 10
    with pytest.raises(ValueError, match="overflows"):
        check_packable(dataset([137.5]), {"v": enc})


def test_unpacked_and_missing_variables_are_ignored():
    ds = dataset([1e9])
    assert (
        check_packable(ds, {"v": {"dtype": "float32"}, "absent": INT16_TENTHS})
        is not None
    )
