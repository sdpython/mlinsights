# -*- coding: utf-8 -*-
"""
@file
@brief Implements a *learner* which follows the same API
as every :epkg:`scikit-learn` learner.
"""
from .sklearn_base import SkBase


class SkBaseLearner(SkBase):

    """
    Pattern of a *learner* qui suit la même API que :epkg:`scikit-learn`.
    """

    def __init__(self, **kwargs):
        """
        constructor
        """
        SkBase.__init__(self, **kwargs)

    ###################
    # API scikit-learn
    ###################

    def fit(self, X, y=None, sample_weight=None):
        """
        Trains a model.

        @param      X               features
        @param      y               targets
        @param      sample_weight   weight
        @return                     self
        """
        raise NotImplementedError()  # pragma: no cover

    def predict(self, X):
        """
        Predicts.

        @param      X   features
        @return         prédictions
        """
        raise NotImplementedError()  # pragma: no cover

    def decision_function(self, X):
        """
        Output of the model in case of a regressor,
        matrix with a score for each class and each sample
        for a classifier.

        @param      X   Samples, {array-like, sparse matrix}, shape = (n_samples, n_features)
        @return         array, shape = (n_samples,.), Returns predicted values.
        """
        raise NotImplementedError()  # pragma: no cover

    def score(self, X, y=None, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels.

        @param      X               Training data, numpy array or sparse matrix of shape [n_samples,n_features]
        @param      y               Target values, numpy array of shape [n_samples, n_targets] (optional)
        @param      sample_weight   Weight values, numpy array of shape [n_samples, n_targets] (optional)
        @return                     score : float, Mean accuracy of self.predict(X) wrt. y.
        """
        raise NotImplementedError()  # pragma: no cover
