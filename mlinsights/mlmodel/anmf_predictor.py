"""
@file
@brief Featurizers for machine learned models.
"""
import numpy
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.decomposition import NMF, TruncatedSVD


class ApproximateNMFPredictor(BaseEstimator, RegressorMixin, MultiOutputMixin):
    """
    Converts :epkg:`sklearn:decomposition:NMF` into
    a predictor so that the prediction does not involve
    training even for new observations. The class uses a
    :epkg:`sklearn:decomposition:TruncatedSVD` of the components
    found by the :epkg:`sklearn:decomposition:NMF`.
    The prediction projects the test data into
    the components vector space and retrieves them back
    into their original space. The issue is it does not
    necessarily produce results with only positive
    results as the :epkg:`sklearn:decomposition:NMF`
    would do unless parameter *force_positive* is True.

    .. runpython::
        :showcode:

        import numpy
        from mlinsights.mlmodel.anmf_predictor import ApproximateNMFPredictor

        train = numpy.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                           [1, 0, 0, 0], [1, 0, 0, 0]], dtype=numpy.float64)
        train[:train.shape[1], :] += numpy.identity(train.shape[1])

        model = ApproximateNMFPredictor(n_components=2,
                                        force_positive=True)
        model .fit(train)

        test = numpy.array([[1, 1, 1, 0]], dtype=numpy.float64)
        pred = model.predict(test)
        print(pred)
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
        self.estimator_svd_ = TruncatedSVD(
            n_components=self.estimator_nmf_.n_components_)
        self.estimator_svd_.fit(self.estimator_nmf_.components_)
        return self

    def predict(self, X):
        """
        Predicts based on the multi-output regressor.
        The output has the same dimension as *X*.
        """
        proj = self.estimator_svd_.transform(X)
        pred = self.estimator_svd_.inverse_transform(proj)
        if self.force_positive:
            zeros = numpy.zeros(
                (1, pred.shape[1]), dtype=pred.dtype)  # pylint: disable=E1101,E1136
            pred = numpy.maximum(pred, zeros)  # pylint: disable=E1111
        return pred
