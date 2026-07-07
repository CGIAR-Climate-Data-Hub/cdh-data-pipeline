"""Berkeley Earth Complete_TAVG example.

Shows multiscale levels, per-level chunking, and int16 scale/offset encoding.

Run from the repo root: uv run recipes/examples/berkeley_tavg.py
"""

import urllib.request
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  registers .rio
import xarray as xr
from zarr.codecs import ZstdCodec
from zarr.codecs.numcodecs import Shuffle

from cdh_data_pipeline import run, write_multiscale_zarr

SRC = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Complete_TAVG_LatLong1.nc"
CACHE = "input/Complete_TAVG_LatLong1.nc"
OUT = "output/examples/berkeley-tavg.zarr"


def fetch():
    if not Path(CACHE).exists():
        Path(CACHE).parent.mkdir(exist_ok=True)
        print(f"  downloading {SRC}")
        urllib.request.urlretrieve(SRC, CACHE)


def build_zarr():
    # rioxarray and topozarr expect x/y spatial dimensions.
    ds = xr.open_dataset(CACHE).rename(longitude="x", latitude="y")
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    ds.attrs.update(
        title="Berkeley Earth surface temperature anomaly (monthly, 1deg)",
        source="Berkeley Earth Surface Temperature Project",
    )
    T = ds.sizes["time"]
    ds = ds[["temperature", "climatology"]]

    codec = (Shuffle(elementsize=2), ZstdCodec(level=19))
    i16 = dict(dtype="int16", _FillValue=-32768, compressors=codec)
    encoding = {
        "temperature": {**i16, "scale_factor": np.float32(0.001)},
        "climatology": {**i16, "scale_factor": np.float32(0.01)},
    }

    def chunking(var, level, sizes):
        yx = (sizes["y"], sizes["x"])
        if var == "climatology":
            return {"chunks": (sizes["month_number"], *yx) if level == 0 else (1, *yx)}
        # Native chunks optimize point time-series reads; overviews optimize map frames.
        if level == 0:
            return {"chunks": (T, 4, 4), "shards": (T, 36, 72)}
        return {"chunks": (1, *yx), "shards": (T, *yx)}

    write_multiscale_zarr(ds, OUT, factors=[2, 4], encoding=encoding, chunking=chunking)


if __name__ == "__main__":
    run(fetch, build_zarr)
