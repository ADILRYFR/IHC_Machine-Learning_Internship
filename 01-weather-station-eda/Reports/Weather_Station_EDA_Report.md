# Weather Station EDA Results

Correlation comparison and measurement visualizations, 01 Jan 2026 to 01 May 2026.

## Correlation Method

- Pearson measures linear relationship between the actual measured values.
- Spearman measures rank-based relationship and is useful when the pattern is monotonic but not perfectly linear.

## Pearson vs Spearman Values

| parameter_1                          | parameter_2                          |   pearson_correlation |   spearman_correlation |   absolute_difference |
|:-------------------------------------|:-------------------------------------|----------------------:|-----------------------:|----------------------:|
| Wind gust (m/s)                      | Wind speed (m/s)                     |             0.969002  |               0.961228 |            0.00777473 |
| Air temperature (C)                  | Pyranometer / solar radiation (W/m2) |             0.556053  |               0.508608 |            0.0474452  |
| Relative humidity (%)                | Wind gust (m/s)                      |            -0.467861  |              -0.495741 |            0.0278797  |
| Pyranometer / solar radiation (W/m2) | Relative humidity (%)                |            -0.459977  |              -0.342933 |            0.117044   |
| Relative humidity (%)                | Wind speed (m/s)                     |            -0.430182  |              -0.459658 |            0.0294754  |
| Air temperature (C)                  | Relative humidity (%)                |            -0.405052  |              -0.40723  |            0.00217855 |
| Pyranometer / solar radiation (W/m2) | Wind gust (m/s)                      |             0.226129  |               0.385777 |            0.159647   |
| Precipitation (mm)                   | Relative humidity (%)                |             0.226826  |               0.336555 |            0.109729   |
| Pyranometer / solar radiation (W/m2) | Wind speed (m/s)                     |             0.16901   |               0.31688  |            0.14787    |
| Barometric pressure (mbar)           | Precipitation (mm)                   |            -0.25841   |              -0.245114 |            0.0132956  |
| Air temperature (C)                  | Wind gust (m/s)                      |             0.0733739 |               0.247821 |            0.174447   |
| Barometric pressure (mbar)           | Relative humidity (%)                |            -0.215126  |              -0.235809 |            0.0206833  |

## Largest Pearson-Spearman Differences

| parameter_1                          | parameter_2           |   pearson_correlation |   spearman_correlation |   absolute_difference |
|:-------------------------------------|:----------------------|----------------------:|-----------------------:|----------------------:|
| Air temperature (C)                  | Wind gust (m/s)       |             0.0733739 |              0.247821  |             0.174447  |
| Pyranometer / solar radiation (W/m2) | Wind gust (m/s)       |             0.226129  |              0.385777  |             0.159647  |
| Air temperature (C)                  | Wind speed (m/s)      |             0.0112922 |              0.169571  |             0.158279  |
| Pyranometer / solar radiation (W/m2) | Wind speed (m/s)      |             0.16901   |              0.31688   |             0.14787   |
| Pyranometer / solar radiation (W/m2) | Relative humidity (%) |            -0.459977  |             -0.342933  |             0.117044  |
| Precipitation (mm)                   | Relative humidity (%) |             0.226826  |              0.336555  |             0.109729  |
| Precipitation (mm)                   | Wind gust (m/s)       |            -0.0197589 |              0.0373799 |             0.0571388 |
| Precipitation (mm)                   | Wind speed (m/s)      |            -0.0187217 |              0.038006  |             0.0567277 |

- See `Weather_Station_EDA_Report.html` for the illustrated report.
- Figures are in `figures/`.
- CSV result tables are in `tables/`.
