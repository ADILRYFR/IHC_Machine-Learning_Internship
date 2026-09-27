# Machine Learning Internship

Weather and PM10 analysis, predictive modelling, and hourly forecasting completed during an internship with International Hellenic University.

## Project guide

| Task | Project | Contents |
| --- | --- | --- |
| 1 | [Weather station EDA](01-weather-station-eda/) | Initial notebook, weather data, correlations and visualisations |
| 2 | [Multi-year weather analysis](02-multi-year-weather-analysis/) | Yearly reports, data quality and cross-year comparisons |
| 3 | [Weather profiles and PM10](03-weather-profiles-and-pm10-analysis/) | Monthly/seasonal profiles and aligned weather-PM10 relationships |
| 4 | [Predictive methods review](04-predictive-methods-review/) | Literature review of regression, tree ensembles and sequence models |
| 5 | [Gradient Boosting](05-gradient-boosting/) | Temperature and PM10 predictions, test results and report |
| 6 | [LSTM neural networks](06-lstm-neural-networks/) | Sequence-based temperature and PM10 predictions |
| 7 | [Model comparison and June forecasts](07-model-comparison-and-june-forecasts/) | Gradient Boosting versus LSTM evaluation and separate forecast CSVs |
| 8 | [Multiple Linear Regression](08-multiple-linear-regression/) | Baseline regression, held-out testing and June forecasts |
| 9 | [NOAA temperature forecasting](09-noaa-temperature-forecasting/) | Standalone three-model study and a 360-hour recursive forecast |

## Reading the results

The local weather/PM10 studies reserve the latest 15 days for chronological testing, not a random 70/30 split. Their test predictions use available measured predictors and historical features. They are not equivalent to a fully unobserved 15-day forecast.

The June CSVs contain forecasts, not verified June observations. No June forecast accuracy is claimed.

Task 9 is a separate temperature-only NOAA study. Its variables and recursive evaluation differ from the earlier local-data studies, so its scores must not be interpreted as evidence that a larger dataset improved or worsened those earlier models. It compares three models within the same NOAA setup. The historical data/test predictions and future 15-day forecast are separate CSV files.

## Running the code

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Each modelling folder contains its required source CSV beside the script. For example:

```sh
python 05-gradient-boosting/train_models.py
```

Running scripts retrains models and may overwrite prediction CSVs. Preserve the submitted results or work in a separate checkout before rerunning. Dependencies are unpinned; this repository does not claim an exact reconstruction of the original software environment. The LSTM implementations use NumPy rather than TensorFlow or PyTorch.

The initial EDA notebook should be opened with its kernel working directory set to `01-weather-station-eda/Source files`. Tasks 2 and 3 preserve analysis reports and tables; no additional executable analysis scripts are supplied for those tasks. HTML reports should be downloaded and opened locally for their full visual presentation.

## Data and attribution

- Local meteorological measurements: university-provided weather-station files.
- PM10 observations: the [IHU meteorological website](http://metro.iem.ihu.gr/).
- NOAA study: [NOAA NCEI](https://www.ncei.noaa.gov/), Thessaloniki/Mikra station `GRMU0016622`.

Data files are retained under their original task-specific names so submitted scripts remain runnable. Some shared source datasets therefore appear in multiple task folders. No blanket licence is assigned to third-party datasets; consult their providers before redistribution or reuse.

This is a curated submission archive, excluding duplicate working submissions, editor caches and the personal internship reflection report. Original report filenames and historical task references are retained for traceability.
