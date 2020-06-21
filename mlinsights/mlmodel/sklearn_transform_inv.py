"""
@file
@brief Implements a base class which defines a pair of transforms
applied around a predictor to modify the target as well.
"""
from sklearn.base import TransformerMixin, BaseEstimator


class BaseReciprocalTransformer(BaseEstimator, TransformerMixin):
    """
    Base for transform which transforms the features
    and the targets at the same time. It must also
    return another transform which transforms the target
    back to what it was.
    """

    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

    def get_fct_inv(self):
        """
        Returns a trained transform which reverse the target
        after a predictor.
        """
        raise NotImplementedError(
            "This must be overwritten.")  # pragma: no cover

    def transform(self, X, y):
        """
        Transforms *X* and *y*.
        Returns transformed *X* and *y*.
        """
        raise NotImplementedError(
            "This must be overwritten.")  # pragma: no cover
