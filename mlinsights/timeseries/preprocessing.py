"""
@file
@brief Timeseries preprocessing.
"""
import numpy
from ..mlmodel.sklearn_transform_inv import BaseReciprocalTransformer


class TimeSeriesDifference(BaseReciprocalTransformer):
    """
    Computes timeseries differences.
    """
    
    def __init__(self, degree=1):
        """
        @param      degree      number of differences
        """
        self.degree = degree
        
    def fit(self, X, y, sample_weight=None):
        """
        Stores the first values.
        """
        self.X_ = X[:self.degree].copy()
        self.y_ = y[:self.degree].copy()
        for n in range(1, self.degree):            
            self.y_[n:] -= self.y_[n-1:-1]
        return self
        
    def transform(self, X, y):
        """
        Transforms both *X* and *y*.
        Returns *X* and *y*.
        """
        for n in range(self.degree):            
            y = y[1:] - y[:-1]
            X = X[1:]
        return X, y

    def get_fct_inv(self):
        """
        Returns the reverse tranform.
        """
        return TimeSeriesDifferenceInv(self).fit()
        
        
class TimeSeriesDifferenceInv(BaseReciprocalTransformer):
    """
    Computes the reverse of @see cl TimeSeriesDifference.
    """
    def __init__(self, estimator):
        """
        @param      estimator   of type @see cl TimeSeriesDifference
        """
        if not isinstance(estimator, TimeSeriesDifference):
            raise TypeError("estimator must be of type TimeSeriesDifference not {}"
                            "".format(type(estimator)))
        self.estimator = estimator
    
    def fit(self, X=None, y=None, sample_weight=None):
        """
        Checks that estimator is fitted.
        """
        if not hasattr(self.estimator, 'X_'):
            raise RuntimeError("Estimator is not fitted.")
        self.estimator_ = self.estimator
        return self

    def transform(self, X, y):
        """
        Transforms both *X* and *y*.
        Returns *X* and *y*.
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
            numpy.cumsum(ny[r0-i-1:, :], axis=0, out=ny[r0-i-1:, :])
        if squeeze:
            numpy.squeeze(ny, out=ny)
        return nx, ny
        
        