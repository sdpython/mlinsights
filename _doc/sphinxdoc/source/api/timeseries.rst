
Timeseries
==========

The following function builds a regular dataset from
a timeseries so that it can be used by machine learning models.

.. autosignature:: mlinsights.timeseries.selection.build_ts_X_y

The first class defined the template for all timeseries
estimators. It deals with a timeseries ine one dimension
and additional features.

.. autosignature:: mlinsights.timeseries.base.BaseTimeSeries

The first regressor is an auto-regressor. It can be estimated
with any regressor implemented in :epkg:`scikit-learn`.

.. autosignature:: mlinsights.timeseries.ar.ARTimeSeriesRegressor
