"""
@file
@brief Auto-regressor for timeseries.
"""
from sklearn.base import RegressorMixin
from .base import BaseTimeSeries
from .selection import check_ts_X_y
from .dummies import DummyTimeSeriesRegressor


class ARTimeSeriesRegressor(BaseTimeSeries, RegressorMixin):
    """
    Base class to build a regressor on timeseries.
    The class computes one or several predictions at each time,
    between *delay1* and *delay2*. It computes:
    :math:`\\hat{Y_{t+d} = f(Y_{t-1}, ..., Y_{t-p})`
    with *d* in *[delay1, delay2[* and
    :math:`1 \\legslant p \\legslant past`.
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
        RegressorMixin.__init__(self)
        BaseTimeSeries.__init__(self, past=past, delay1=delay1, delay2=delay2,
                                use_all_past=use_all_past)
        if estimator == "dummy":
            self.estimator = DummyTimeSeriesRegressor(
                past=past, delay1=delay1, delay2=delay2, use_all_past=use_all_past)
        if not hasattr(self.estimator, "fit"):
            raise TypeError(
                "estimator is not an estimator but {}".format(type(estimator)))

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
        self.estimator_ = (self.estimator.fit(X, y)
                           if sample_weight is None
                           else self.estimator.fit(X, y, sample_weight=sample_weight))
        return self

    def predict(self, X):
        """
        Returns the prediction
        """
        check_ts_X_y(self, X, None)
        return self.estimator_.predict(X)
