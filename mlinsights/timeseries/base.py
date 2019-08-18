"""
@file
@brief Base class for timeseries.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from .metrics import ts_mape


class BaseTimeSeries(BaseEstimator):
    """
    Base class to build a predictor on timeseries.
    The class computes one or several predictions at each time,
    between *delay1* and *delay2*. It computes:
    :math:`\\hat{Y_{t+d} = f(Y_{t-1}, ..., Y_{t-p})`
    with *d* in *[delay1, delay2[* and
    :math:`1 \\legslant p \\legslant past`.
    """

    def __init__(self, past=7, delay1=1, delay2=2, use_all_past=False):
        """
        @param      past            values to use to predict
        @param      delay1          the model computes the first prediction for
                                    *time=t + delay1*
        @param      delay2          the model computes the last prediction for
                                    *time=t + delay2* excluded
        @param      use_all_past    use all past features, not only the timeseries
        """
        self.past = past
        self.delay1 = delay1
        self.delay2 = delay2
        self.use_all_past = use_all_past
        if self.delay1 < 1:
            raise ValueError("delay1 must be >= 1")
        if self.delay2 <= self.delay1:
            raise ValueError("delay2 must be >= 1")
        if self.past < 0:
            raise ValueError("past must be > 0")


class TimeSeriesRegressorMixin(RegressorMixin):
    """
    Addition to :epkg:`sklearn:base:RegressorMixin`.
    """

    def score(self, X, y, sample_weight=None):
        """
        Scores the prediction using
        @see fn ts_mape

        :param X: features
        :param y: expected values
        :param sample_weight: sample weight
        :return: see @see fn ts_mape
        """
        pred = self.predict(X)
        return ts_mape(y, pred, sample_weight=sample_weight)
