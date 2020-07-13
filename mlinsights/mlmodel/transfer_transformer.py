"""
@file
@brief Implements a transformer which wraps a predictor
to do transfer learning.
"""
import inspect
from sklearn.base import BaseEstimator, TransformerMixin
from .sklearn_testing import clone_with_fitted_parameters


class TransferTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps a predictor or a transformer in a transformer.
    This model is frozen: it cannot be trained and only
    computes the predictions.

    .. index:: transfer learning, frozen model
    """

    def __init__(self, estimator, method=None, copy_estimator=True,
                 trainable=False):
        """
        @param      estimator           estimator to wrap in a transformer, it is cloned
                                        with the training data (deep copy) when fitted
        @param      method              if None, guess what method should be called,
                                        *transform* for a transformer,
                                        *predict_proba* for a classifier,
                                        *decision_function* if found,
                                        *predict* otherwiser
        @param      copy_estimator      copy the model instead of taking a reference
        @param      trainable           the transfered model must be trained
        """
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.estimator = estimator
        self.copy_estimator = copy_estimator
        self.trainable = trainable
        if method is None:
            if hasattr(estimator, "transform"):
                method = "transform"
            elif hasattr(estimator, "predict_proba"):
                method = "predict_proba"
            elif hasattr(estimator, "decision_function"):
                method = "decision_function"
            elif hasattr(estimator, "predict"):
                method = "predict"
            else:
                raise AttributeError(  # pragma: no cover
                    "Cannot find a method transform, predict_proba, decision_function, "
                    "predict in object {}".format(type(estimator)))
        if not hasattr(estimator, method):
            raise AttributeError(  # pragma: no cover
                "Cannot find method '{}' in object {}".format(
                    method, type(estimator)))
        self.method = method

    def fit(self, X=None, y=None, sample_weight=None):
        """
        The function does nothing.

        :param X: unused
        :param y: unused
        :param sample_weight: unused
        :return: self: returns an instance of self.

        Fitted attributes:

        * `estimator_`: already trained estimator
        """
        if self.copy_estimator:
            self.estimator_ = clone_with_fitted_parameters(self.estimator)
            from .sklearn_testing import assert_estimator_equal  # pylint: disable=C0415
            assert_estimator_equal(self.estimator_, self.estimator)
        else:
            self.estimator_ = self.estimator
        if self.trainable:
            insp = inspect.signature(self.estimator_.fit)
            pars = insp.parameters
            if 'y' in pars and 'sample_weight' in pars:
                self.estimator_.fit(X, y, sample_weight)
            elif 'y' in pars:
                self.estimator_.fit(X, y)
            elif 'sample_weight' in pars:
                self.estimator_.fit(X, sample_weight=sample_weight)
            else:
                self.estimator_.fit(X)
        return self

    def transform(self, X):
        """
        Runs the predictions.

        :param X: numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        :return: tranformed *X*
        """
        meth = getattr(self.estimator_, self.method)
        return meth(X)
