"""
@file
@brief Dummy auto-regressor which takes past values as predictions.
"""
import numpy
from .base import BaseTimeSeries, TimeSeriesRegressorMixin
from .selection import check_ts_X_y


class DummyTimeSeriesRegressor(BaseTimeSeries, TimeSeriesRegressorMixin):
    """
    Dummy regressor for time series. Use past values as prediction.
    """

    def __init__(self, estimator="dummy", past=7, delay1=1, delay2=2, use_all_past=False):
        """
        @param      estimator       estimator to use for regression,
                                    :epkg:`sklearn:linear_model:LinearRegression`
                                    implements a linear auto-regressor,
                                    ``'dummy'`` use past value as predictions
        @param      past            values to use to predict
        @param      delay1          the model computes the first prediction for
                                    *time=t + delay1*
        @param      delay2          the model computes the last prediction for
                                    *time=t + delay2* excluded
        @param      use_all_past    use all past features, not only the timeseries
        """
        TimeSeriesRegressorMixin.__init__(self)
        BaseTimeSeries.__init__(self, past=past, delay1=delay1, delay2=delay2,
                                use_all_past=use_all_past)

    def fit(self, X, y, sample_weight=None):
        """
        Trains the model.

        Parameters
        ----------

        X: output of
            X may be empty (None)
        y: timeseries (one single vector), array [n_obs]
        sample_weight: weights None or array [n_obs]

        Returns
        -------

        self
        """
        check_ts_X_y(self, X, y)
        return self

    def predict(self, X):
        """
        Returns the prediction
        """
        check_ts_X_y(self, X, None)
        nb = self.delay2 - self.delay1
        pred = numpy.empty((X.shape[0], nb), dtype=X.dtype)
        for i in range(0, nb):
            pred[:, i] = X[:, -1]
        return pred
