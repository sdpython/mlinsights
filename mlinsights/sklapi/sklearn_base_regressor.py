# -*- coding: utf-8 -*-
"""
@file
@brief Implements @see cl SkBaseRegressor.
"""
from sklearn.metrics import r2_score
from .sklearn_base_learner import SkBaseLearner


class SkBaseRegressor(SkBaseLearner):
    """
    Defines a custom regressor.
    """

    def __init__(self, **kwargs):
        """
        constructor
        """
        SkBaseLearner.__init__(self, **kwargs)

    def score(self, X, y=None, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels.

        @param      X               Training data, numpy array or sparse matrix of shape [n_samples,n_features]
        @param      y               Target values, numpy array of shape [n_samples, n_targets] (optional)
        @param      sample_weight   Weight values, numpy array of shape [n_samples, n_targets] (optional)
        @return                     score : float, Mean accuracy of self.predict(X) wrt. y.
        """
        return r2_score(y, self.predict(X), sample_weight=sample_weight)
