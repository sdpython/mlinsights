"""
@file
@brief Implements a quantile linear regression.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from .sklearn_testing import clone_with_fitted_parameters


class TransferTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps a predictor or a transformer in a transformer.
    This model is frozen: it cannot be trained and only
    computes the predictions.

    .. index:: transfer learning, frozen model
    """

    def __init__(self, estimator, method=None, copy_estimator=True):
        """
        @param      estimator           estimator to wrap in a transformer, it is cloned
                                        with the training data (deep copy) when fitted
        @param      method              if None, guess what method should be called,
                                        *transform* for a transformer,
                                        *predict_proba* for a classifier,
                                        *decision_function* if found,
                                        *predict* otherwiser
        @param      copy_estimator      copy the model instead of taking a reference
        """
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.estimator = estimator
        self.copy_estimator = copy_estimator
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
                raise AttributeError(
                    "Cannot find a method transform, predict_proba, decision_function, predict in object {}".format(type(estimator)))
        if not hasattr(estimator, method):
            raise AttributeError(
                "Cannot find method '{}' in object {}".format(method, type(estimator)))
        self.method = method

    def fit(self, X, y=None, sample_weight=None):
        """
        The function does nothing.

        Parameters
        ----------
        X : unused

        y : unused

        sample_weight : unused

        Returns
        -------
        self : returns an instance of self.

        Attributes
        ----------

        estimator_: already trained estimator
        """
        if self.copy_estimator:
            self.estimator_ = clone_with_fitted_parameters(self.estimator)
            from .sklearn_testing import assert_estimator_equal
            assert_estimator_equal(self.estimator_, self.estimator)
        else:
            self.estimator_ = self.estimator
        return self

    def transform(self, X):
        """
        Runs the predictions.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        Returns
        -------
        tranformed *X*
        """
        meth = getattr(self.estimator_, self.method)
        return meth(X)
