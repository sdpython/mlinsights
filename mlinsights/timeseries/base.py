"""
@file
@brief Base class for timeseries.
"""
import numpy
from sklearn.base import BaseEstimator


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

    def fit(self, X, y, weights=None):
        """
        Trains the model.

        Parameters
        ----------

        X: other times series, used as features, array [n_obs, n_features],
            X may be empty (None)
        y: timeseries (one single vector), array [n_obs]
        weights: weights None or array [n_obs]

        Returns
        -------

        self
        """

    def build_X_y(self, X, y, weights=None):
        """
        Builds standard *X, y* based in the given one.

        Parameters
        ----------

        X: other times series, used as features, [n_obs, n_features],
            X may be empty (None)
        y: timeseries (one single vector),  [n_obs]
        weights: weights None or array [n_obs]

        Returns
        -------
        X: array of features [nrows, n_features + past]
            where `nrows = n_obs + self.delay2 - self.past + 2`
        y: array of targets [nrows]
        weights: None or array [nrows]

        A few examples.

        .. runpython::
            :showcode:

            import numpy
            from mlinsights.timeseries.base import BaseTimeSeries

            X = numpy.arange(10).reshape(5, 2)
            y = numpy.arange(5) * 100
            weights = numpy.arange(5) * 1000
            bs = BaseTimeSeries(past=2)
            nx, ny, nw = bs.build_X_y(X, y, weights)
            print('X=', X)
            print('y=', y)
            print('nx=', nx)
            print('ny=', ny)

        With ``use_all_past=True``:

        .. runpython::
            :showcode:

            import numpy
            from mlinsights.timeseries.base import BaseTimeSeries

            X = numpy.arange(10).reshape(5, 2)
            y = numpy.arange(5) * 100
            weights = numpy.arange(5) * 1000
            bs = BaseTimeSeries(past=2, use_all_past=True)
            nx, ny, nw = bs.build_X_y(X, y, weights)
            print('X=', X)
            print('y=', y)
            print('nx=', nx)
            print('ny=', ny)
        """
        if self.use_all_past:
            ncol = X.shape[1]
            nrow = y.shape[0] - self.delay2 - self.past + 2
            new_X = numpy.empty(
                (nrow, ncol * self.past + self.past), dtype=y.dtype)
            for i in range(0, self.past):
                begin = i * ncol
                end = begin + ncol
                new_X[:, begin:end] = X[i: i + nrow]
            for i in range(0, self.past):
                end = y.shape[0] + i + self.delay1 - 1 - self.delay2
                new_X[:, i + ncol * self.past] = y[i: end]
            new_y = numpy.empty(
                (nrow, self.delay2 - self.delay1), dtype=y.dtype)
            for i in range(self.delay1, self.delay2):
                new_y[:, i - self.delay1] = y[i + 1:i + nrow + 1]
            new_weights = (None if weights is None
                           else weights[self.past - 1:self.past - 1 + nrow])
        else:
            ncol = X.shape[1]
            nrow = y.shape[0] - self.delay2 - self.past + 2
            new_X = numpy.empty((nrow, ncol + self.past), dtype=y.dtype)
            new_X[:, :X.shape[1]] = X[self.past -
                                      1: X.shape[0] - self.delay2 + 1]
            for i in range(self.past):
                end = y.shape[0] + i + self.delay1 - 1 - self.delay2
                new_X[:, i + ncol] = y[i: end]
            new_y = numpy.empty(
                (nrow, self.delay2 - self.delay1), dtype=y.dtype)
            for i in range(self.delay1, self.delay2):
                new_y[:, i - self.delay1] = y[i + 1:i + nrow + 1]
            new_weights = (None if weights is None
                           else weights[self.past - 1:self.past - 1 + nrow])
        return new_X, new_y, new_weights