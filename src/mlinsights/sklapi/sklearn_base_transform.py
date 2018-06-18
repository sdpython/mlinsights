# -*- coding: utf-8 -*-
"""
@file
@brief Implements a *transform* which follows the smae API
as every :epkg:`scikit-learn` transform.
"""
from .sklearn_base import SkBase


class SkBaseTransform(SkBase):
    """
    Pattern of a *learner* which follows the same API que :epkg:`scikit-learn`.
    """

    def __init__(self, **kwargs):
        """
        Stores the parameters.
        """
        SkBase.__init__(self, **kwargs)

    ###################
    # API scikit-learn
    ###################

    def fit(self, X, y=None, **kwargs):
        """
        Trains a model.

        @param      X               features
        @param      y               targets
        @return                     self
        """
        raise NotImplementedError()

    def transform(self, X):
        """
        Transforms the data.

        @param      X   features
        @return         predictions
        """
        raise NotImplementedError()

    def fit_transform(self, X, y=None, **kwargs):
        """
        Trains and transforms the data.

        @param      X               features
        @param      y               targets
        @return                     self
        """
        self.fit(X, y=y, **kwargs)
        return self.transform(X)
