# -*- coding: utf-8 -*-
"""
@file
@brief Implements class @see cl SkBaseClassifier.
"""
from sklearn.metrics import accuracy_score
from .sklearn_base_learner import SkBaseLearner


class SkBaseClassifier(SkBaseLearner):
    """
    Defines a custom classifier.
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
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def predict_proba(self, X):
        """
        Returns probability estimates for the test data X.

        @param      X       Training data, numpy array or sparse matrix of shape [n_samples,n_features]
        @return             array, shape = (n_samples,.), Returns predicted values.
        """
        raise NotImplementedError()
