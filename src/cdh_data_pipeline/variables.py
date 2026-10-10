"""Concrete daily variables, one entry per array a mission writes.

Quantum choice, measured on eight days of CHIRTS-ERA5 (source 26.40 MB/day):

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

__all__ = ["TMAX", "TMIN"]
