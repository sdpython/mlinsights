
Timeseries
==========

.. contents::
    :local:

Datasets
++++++++

.. autofunction:: mlinsights.timeseries.datasets.artificial_data

Experimentation
+++++++++++++++

.. autofunction:: mlinsights.timeseries.patterns.find_ts_group_pattern

Manipulation
++++++++++++

.. autofunction:: mlinsights.timeseries.agg.aggregate_timeseries

Plotting
++++++++

.. autofunction:: mlinsights.timeseries.plotting.plot_week_timeseries

Prediction
++++++++++

The following function builds a regular dataset from
a timeseries so that it can be used by machine learning models.

.. autofunction:: mlinsights.timeseries.utils.build_ts_X_y

The first class defined the template for all timeseries
estimators. It deals with a timeseries ine one dimension
and additional features.

.. autoclass:: mlinsights.timeseries.base.BaseTimeSeries
    :members:

the first predictor is a dummy one: it uses the current value to
predict the future.

.. autoclass:: mlinsights.timeseries.dummies.DummyTimeSeriesRegressor
    :members:

The first regressor is an auto-regressor. It can be estimated
with any regressor implemented in :epkg:`scikit-learn`.

.. autoclass:: mlinsights.timeseries.ar.ARTimeSeriesRegressor
    :members:

The library implements one scoring function which compares
the prediction to what a dummy predictor would do
by using the previous day as a prediction.

.. autofunction:: mlinsights.timeseries.metrics.ts_mape
