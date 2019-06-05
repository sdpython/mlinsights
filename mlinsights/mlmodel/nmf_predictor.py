"""
@file
@brief Featurizers for machine learned models.
"""
import numpy
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.decomposition import NMF, TruncatedSVD


class NMFPredictor(BaseEstimator, RegressorMixin, MultiOutputMixin):
    """
    Converts :epkg:`sklearn:decomposition:NMF` into
    a predictor so that the prediction does not involve
    training even for new observations. The class uses a
    :epkg:`sklearn:decomposition:TruncatedSVD` of the components
    found by the :epkg:`sklearn:decomposition:NMF`.
    The prediction projects the test data into
    the components vector space and retrieves them back
    into their original space. The issue is it does not
    necessarily produces results with only positive
    results as the :epkg:`sklearn:decomposition:NMF`
    would do.
    """

    def __init__(self, force_positive=False, **kwargs):
        """
        *kwargs* should contains parameters
        for :epkg:`sklearn:decomposition:NMF`.
        The parameter *force_positive* removes all
        negative predictions and replaces by zero.
        """
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)
        MultiOutputMixin.__init__(self)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.force_positive = force_positive

    @classmethod
    def _get_param_names(cls):
        """
        Returns the list of parameters
        of the estimator.
        """
        res = NMF._get_param_names()
        res = res + ["force_positive"]
        return res

    def get_params(self, deep=True):
        """
        Returns the parameters of the estimator
        as a dictionary.
        """
        res = {}
        for k in self.__class__._get_param_names():
            if hasattr(self, k):
                res[k] = getattr(self, k)
        return res

    def fit(self, X, y=None):
        """
        Trains a :epkg:`sklearn:decomposition:NMF`
        then a multi-output regressor.
        """
        params = self.get_params()
        if 'force_positive' in params:
            del params['force_positive']
        self.estimator_nmf_ = NMF(**params)
        self.estimator_nmf_.fit(X)
        self.estimator_svd_ = TruncatedSVD(n_components=self.estimator_nmf_.n_components_)
        self.estimator_svd_.fit(self.estimator_nmf_.components_)
        return self

    def predict(self, X):
        """
        Predicts based on the multi-output regressor.
        """
        proj = self.estimator_svd_.transform(X)
        pred = self.estimator_svd_.inverse_transform(proj)
        if self.force_positive:
            zeros = numpy.zeros((1, pred.shape[1]), dtype=pred.dtype)
            pred = numpy.maximum(pred, zeros)
        return pred
