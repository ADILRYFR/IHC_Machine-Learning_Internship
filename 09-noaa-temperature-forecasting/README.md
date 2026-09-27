# NOAA Temperature Forecasting: Three-Model Comparison

Compare Multiple Linear Regression, Gradient Boosting and LSTM in a standalone recursive hourly temperature study.

## Files and execution

Run `python 09-noaa-temperature-forecasting/task9_recursive_three_model.py`. The historical source/test file is `task9_noaa_data_and_predictions.csv`; the future output is `task9_15_day_forecast.csv`. Running the script retrains models and rewrites these outputs. The 15-day horizon contains 360 hourly timestamps. This study is not directly comparable with the local weather/PM10 tasks.

Install the repository's root requirements before running code. Original report filenames and task numbers are preserved.
