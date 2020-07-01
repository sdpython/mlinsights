"""
@file
@brief Timeseries preprocessing.
"""
import numpy
from .base import BaseReciprocalTimeSeriesTransformer


class TimeSeriesDifference(BaseReciprocalTimeSeriesTransformer):
    """
    Computes timeseries differences.
    """

    def __init__(self, degree=1):
        """
        @param      degree      number of differences
        """
        BaseReciprocalTimeSeriesTransformer.__init__(self, degree)

    @property
    def degree(self):
        """
        Returns the degree.
        """
        return self.context_length

    def fit(self, X, y, sample_weight=None):
        """
        Stores the first values.
        """
        self.X_ = X[:self.degree].copy()
        self.y_ = y[:self.degree].copy()
        for n in range(1, self.degree):
            self.y_[n:] -= self.y_[n - 1:-1]
        return self

    def transform(self, X, y, sample_weight=None):
        """
        Transforms both *X* and *y*.
        Returns *X* and *y*, returns
        *sample_weight* as well if not None.
        """
        for _ in range(self.degree):
            y = y[1:] - y[:-1]
            X = X[1:]
        if sample_weight is None:
            return X, y
        return X, y, sample_weight[1:]

    def get_fct_inv(self):
        """
        Returns the reverse tranform.
        """
        return TimeSeriesDifferenceInv(self).fit()


class TimeSeriesDifferenceInv(BaseReciprocalTimeSeriesTransformer):
    """
    Computes the reverse of @see cl TimeSeriesDifference.
    """

    def __init__(self, estimator):
        """
        @param      estimator   of type @see cl TimeSeriesDifference
        """
        BaseReciprocalTimeSeriesTransformer.__init__(
            self, estimator.context_length)
        if not isinstance(estimator, TimeSeriesDifference):
            raise TypeError(  # pragma: no cover
                "estimator must be of type TimeSeriesDifference not {}"
                "".format(type(estimator)))
        self.estimator = estimator

    def fit(self, X=None, y=None, sample_weight=None):
        """
        Checks that estimator is fitted.
        """
        if not hasattr(self.estimator, 'X_'):
            raise RuntimeError(  # pragma: no cover
                "Estimator is not fitted.")
        self.estimator_ = self.estimator
        return self

    def transform(self, X, y, sample_weight=None):
        """
        Transforms both *X* and *y*.
        Returns *X* and *y*, returns
        *sample_weight* as well if not None.
        """
        if len(y.shape) == 1:
            y = y.reshape((y.shape[0], 1))
            squeeze = True
        else:
            squeeze = False
        if len(self.estimator_.y_.shape) == 1:
            y0 = self.estimator_.y_.reshape((y.shape[0], 1))
        else:
            y0 = self.estimator_.y_
        r0 = self.estimator_.X_.shape[0]

        nx = numpy.empty((r0 + X.shape[0], X.shape[1]), dtype=X.dtype)
        nx[:r0, :] = self.estimator_.X_
        nx[r0:, :] = X

        ny = numpy.empty((r0 + X.shape[0], y.shape[1]), dtype=X.dtype)
        ny[:r0, :] = y0
        ny[r0:, :] = y

        for i in range(self.estimator_.degree):
            numpy.cumsum(ny[r0 - i - 1:, :], axis=0, out=ny[r0 - i - 1:, :])
        if squeeze:
            ny = numpy.squeeze(ny)
        if sample_weight is None:
            return nx, ny
        nw = numpy.zeros(ny.shape[0])
        de = nw.shape[0] - sample_weight.shape[0]
        nw[de:] = sample_weight
        return nx, ny, nw
