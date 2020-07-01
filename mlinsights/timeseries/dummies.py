"""
@file
@brief Dummy auto-regressor which takes past values as predictions.
"""
import numpy
from .base import BaseTimeSeries, TimeSeriesRegressorMixin
from .utils import check_ts_X_y


class DummyTimeSeriesRegressor(BaseTimeSeries, TimeSeriesRegressorMixin):
    """
    Dummy regressor for time series. Use past values as prediction.
    """

    def __init__(self, estimator="dummy", past=1, delay1=1, delay2=2,
                 use_all_past=False, preprocessing=None):
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
        @param      preprocessing   preprocessing to apply before predicting,
                                    only the timeseries itselves, it can be
                                    a difference, it must be of type
                                    @see cl BaseReciprocalTimeSeriesTransformer
        """
        TimeSeriesRegressorMixin.__init__(self)
        BaseTimeSeries.__init__(self, past=past, delay1=delay1, delay2=delay2,
                                use_all_past=use_all_past, preprocessing=preprocessing)

    def fit(self, X, y, sample_weight=None):
        """
        Trains the model.

        :param X: output of
            X may be empty (None)
        :param y: timeseries (one single vector), array [n_obs]
        :param sample_weight: weights None or array [n_obs]
        :return: self
        """
        X, y, _ = self._base_fit_predict(X, y, sample_weight)
        check_ts_X_y(self, X, y)
        return self

    def predict(self, X, y):
        """
        Returns the prediction
        """
        X, y, _ = self._base_fit_predict(X, y, None)
        check_ts_X_y(self, X, y)
        nbrow = X.shape[0]
        X, y = self._applies_preprocessing(X, y, None)[:2]
        nb = self.delay2 - self.delay1
        pred = numpy.empty((nbrow, nb), dtype=X.dtype)
        first = nbrow - X.shape[0]
        pred[:first] = numpy.nan
        for i in range(0, nb):
            pred[first:, i] = X[:, -1]
        return pred
