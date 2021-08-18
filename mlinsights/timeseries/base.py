"""
@file
@brief Base class for timeseries.
"""
from sklearn.base import BaseEstimator, RegressorMixin, clone
from ..mlmodel.sklearn_transform_inv import BaseReciprocalTransformer
from .metrics import ts_mape
from .utils import check_ts_X_y, build_ts_X_y


class BaseReciprocalTimeSeriesTransformer(BaseReciprocalTransformer):
    """
    Base for all timeseries preprocessing
    automatically applied within a predictor.
    """

    def __init__(self, context_length=0):
        """
        @param      context_length  number of previous observations to
                                    build or rebuild the observations
        """
        BaseReciprocalTransformer.__init__(self)
        self.context_length = context_length

    def fit(self, X, y, sample_weight=None):
        """
        Stores the first values.
        """
        raise NotImplementedError("Should be overwritten.")  # pragma: no cover

    def transform(self, X, y, sample_weight=None, context=None):
        """
        Transforms both *X* and *y*.
        Returns *X* and *y*, returns
        *sample_weight* as well if not None.
        The context is used when the *y* series stored
        in the predictor is not related to the *y* series
        given to the *transform* method.
        """
        raise NotImplementedError("Should be overwritten.")  # pragma: no cover

    def get_fct_inv(self):
        """
        Returns the reverse tranform.
        """
        raise NotImplementedError("Should be overwritten.")  # pragma: no cover


class BaseTimeSeries(BaseEstimator):
    """
    Base class to build a predictor on timeseries.
    The class computes one or several predictions at each time,
    between *delay1* and *delay2*. It computes:
    :math:`\\hat{Y_{t+d} = f(Y_{t-1}, ..., Y_{t-p})}`
    with *d* in *[delay1, delay2[* and
    :math:`1 \\leqslant p \\leqslant past`.
    """

    def __init__(self, past=1, delay1=1, delay2=2,
                 use_all_past=False, preprocessing=None):
        """
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
        self.past = past
        self.delay1 = delay1
        self.delay2 = delay2
        self.use_all_past = use_all_past
        self.preprocessing = preprocessing
        if self.delay1 < 1:
            raise ValueError("delay1 must be >= 1")  # pragma: no cover
        if self.delay2 <= self.delay1:
            raise ValueError("delay2 must be >= 1")  # pragma: no cover
        if self.past < 0:
            raise ValueError("past must be > 0")  # pragma: no cover
        if (preprocessing is not None and
                not isinstance(preprocessing, BaseReciprocalTimeSeriesTransformer)):
            raise TypeError(  # pragma: no cover
                "preprocessing must be of type 'BaseReciprocalTimeSeriesTransformer' "
                "not {}".format(type(preprocessing)))

    def _fit_preprocessing(self, X, y, sample_weight=None):
        """
        Applies the preprocessing.
        *X*, *y*, *sample_weight*.

        :param X: output of
            X may be empty (None)
        :param y: timeseries (one single vector), array [n_obs]
        :param sample_weight: weights None or array [n_obs]
        :return: *X*, *y*, *sample_weight*
        """
        check_ts_X_y(self, X, y)

        if self.preprocessing is not None:
            self.preprocessing_ = clone(self.preprocessing)
            self.preprocessing_.fit(X, y, sample_weight)
            xyw = self.preprocessing_.transform(X, y, sample_weight)
            X, y = xyw[:2]
            sample_weight = xyw[-1] if sample_weight is not None else None
        return X, y, sample_weight

    def _base_fit_predict(self, X, y, sample_weight=None):
        """
        Trains the preprocessing and returns the modified
        *X*, *y*, *sample_weight*.

        :param X: output of
            X may be empty (None)
        :param y: timeseries (one single vector), array [n_obs]
        :param sample_weight: weights None or array [n_obs]
        :return: *X*, *y*, *sample_weight*

        The *y* series is moved by *self.delay1* in the past.
        """
        if y is None:
            raise RuntimeError("y cannot be None")  # pragma: no cover
        X, y, sample_weight = build_ts_X_y(
            self, X, y, sample_weight, same_rows=True)
        X, y, sample_weight = self._fit_preprocessing(X, y, sample_weight)
        return X, y, sample_weight

    def has_preprocessing(self):
        """
        Tells if there is one preprocessing.
        """
        return hasattr(self, 'preprocessing_') and self.preprocessing_ is not None

    def _applies_preprocessing(self, X, y, sample_weight):
        """
        Applies the preprocessing to the series.
        """
        if self.has_preprocessing():
            xyw = self.preprocessing_.transform(X, y, sample_weight)
            X, y = xyw[:2]
            sample_weight = xyw[-1] if sample_weight is not None else None
        return X, y, sample_weight

    def _applies_preprocessing_inv(self, X, y, sample_weight):
        """
        Applies the preprocessing to the series.
        """
        if self.has_preprocessing():
            inv = self.preprocessing_.get_fct_inv()
            X, y, sample_weight = inv.transform(X, y, sample_weight)

        return X, y, sample_weight


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
        pred = self.predict(X, y)
        return ts_mape(y, pred, sample_weight=sample_weight)
