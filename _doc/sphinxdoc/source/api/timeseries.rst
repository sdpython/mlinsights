
Timeseries
==========

.. contents::
    :local:

Datasets
++++++++

.. autosignature:: mlinsights.timeseries.datasets.artificial_data

Experimentation
+++++++++++++++

.. autosignature:: mlinsights.timeseries.patterns.find_ts_group_pattern

Manipulation
++++++++++++

.. autosignature:: mlinsights.timeseries.agg.aggregate_timeseries

Plotting
++++++++

.. autosignature:: mlinsights.timeseries.plotting.plot_week_timeseries

Prediction
++++++++++

The following function builds a regular dataset from
a timeseries so that it can be used by machine learning models.

.. autosignature:: mlinsights.timeseries.selection.build_ts_X_y

The first class defined the template for all timeseries
estimators. It deals with a timeseries ine one dimension
and additional features.

.. autosignature:: mlinsights.timeseries.base.BaseTimeSeries

the first predictor is a dummy one: it uses the current value to
predict the future.

.. autosignature:: mlinsights.timeseries.dummies.DummyTimeSeriesRegressor

The first regressor is an auto-regressor. It can be estimated
with any regressor implemented in :epkg:`scikit-learn`.

.. autosignature:: mlinsights.timeseries.ar.ARTimeSeriesRegressor

The library implements one scoring function which compares
the prediction to what a dummy predictor would do
by using the previous day as a prediction.

.. autosignature:: mlinsights.timeseries.metrics.ts_mape
