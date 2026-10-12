"""Concrete daily variables, one entry per array a mission writes.

Temperature quantum, measured on eight days of CHIRTS-ERA5 (source 26.40 MB/day):

==========  =========  ==========  ========
quantum       MB/day     vs 0.01    max err
==========  =========  ==========  ========
0.01 C           6.16        100%    0.005 C
**0.1 C**    **3.52**     **57%**    0.05 C
0.2 C            2.73         44%     0.1 C
==========  =========  ==========  ========

0.1 C is a 7.5x reduction against the source. CHIRTS-daily's own validation
reports a mean absolute error of 1-2 C against station observations, so a 0.05 C
rounding error sits 20-40x inside the product's uncertainty. Finer settings buy
noise.
"""

from cdh_data_pipeline.utils import MeteoVariable

#: Earth's records are roughly -90..57 C. The bounds flag corruption, not merely
#: unusual weather.
_T_VALID_MIN = -100.0
_T_VALID_MAX = 70.0

TMAX = MeteoVariable(
    key="tmax",
    nc_name="Tmax",
    folder="tmax_daily",
    dtype="int16",
    scale=0.1,
    offset=0.0,
    fill=-32768,
    units="degC",
    long_name="Daily maximum 2 m air temperature",
    standard_name="air_temperature",
    valid_min=_T_VALID_MIN,
    valid_max=_T_VALID_MAX,
)

TMIN = MeteoVariable(
    key="tmin",
    nc_name="Tmin",
    folder="tmin_daily",
    dtype="int16",
    scale=0.1,
    offset=0.0,
    fill=-32768,
    units="degC",
    long_name="Daily minimum 2 m air temperature",
    standard_name="air_temperature",
    valid_min=_T_VALID_MIN,
    valid_max=_T_VALID_MAX,
)

#: CHIRPS v3.0 daily precipitation, read from the daily GeoTIFF archive.
#:
#: ``folder`` is empty because the archive has no per-variable level: rasters sit
#: straight under ``<root>/<year>/``.
#:
#: ``fill`` is -9999 rather than int16's minimum because that is what the
#: published cube already stores; changing it would invalidate every chunk
#: written so far. Note the consequence: -9999 sits *inside* the int16 range, so
#: unlike the temperature variables it does not shrink :attr:`store_bounds`.
PRECIP = MeteoVariable(
    key="precip",
    nc_name="precip",
    folder="",
    dtype="int16",
    scale=0.1,
    offset=0.0,
    fill=-9999,
    units="mm",
    long_name="CHIRPS v3.0 daily precipitation",
    standard_name="precipitation_amount",
    # CHIRPS daily maxima are observed around 750 mm; the int16 ceiling at this
    # quantum is 3276 mm, so the bound flags corruption, not heavy rain.
    valid_min=0.0,
    valid_max=3276.0,
)

__all__ = ["PRECIP", "TMAX", "TMIN"]
