# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""

import sys
import os
import unittest
import numpy
from numpy.random import RandomState
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from pyquickhelper.pycode import ExtTestCase

try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from src.mlinsights.mlmodel import PredictableTSNE
from src.mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone


class TestPredictableTSNE(ExtTestCase):

    def test_predictable_tsne(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        clr = PredictableTSNE()
        memoize_targets = []
        clr.fit(X, y, memoize_targets=memoize_targets)
        pred = clr.transform(X)
        diff = numpy.trace(pred.T @ memoize_targets[0]) / X.shape[0]
        self.assertGreater(diff, 0)
        self.assertIsInstance(clr.estimator_, MLPRegressor)

    def test_predictable_tsne_knn(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        clr = PredictableTSNE(estimator=KNeighborsRegressor())
        memoize_targets = []
        clr.fit(X, y, memoize_targets=memoize_targets)
        pred = clr.transform(X)
        diff = numpy.trace(pred.T @ memoize_targets[0]) / X.shape[0]
        self.assertGreater(diff, 0)
        self.assertIsInstance(clr.estimator_, KNeighborsRegressor)

    def test_predictable_tsne_intercept_weights(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        clr = PredictableTSNE()
        memoize_targets = []
        clr.fit(X, y, sample_weight=numpy.ones((X.shape[0],)),
                memoize_targets=memoize_targets)
        acc = clr.transform(X)
        diff = numpy.trace(acc.T @ memoize_targets[0]) / X.shape[0]
        self.assertGreater(diff, 0)

    def test_predictable_tsne_pickle(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        test_sklearn_pickle(lambda: PredictableTSNE(), X, y)

    def test_predictable_tsne_clone(self):
        self.maxDiff = None
        test_sklearn_clone(lambda: PredictableTSNE())

    def test_predictable_tsne_relevance(self):
        state = RandomState(seed=0)
        Xs = []
        Ys = []
        n = 20
        for i in range(0, 5):
            for j in range(0, 4):
                x1 = state.rand(n) + i * 1.1
                x2 = state.rand(n) + j * 1.1
                Xs.append(numpy.vstack([x1, x2]).T)
                cl = state.randint(0, 4)
                Ys.extend([cl for i in range(n)])
        X = numpy.vstack(Xs)
        Y = numpy.array(Ys)
        clk = PredictableTSNE(t_n_components=3, normalizer=StandardScaler())
        memoize_targets = []
        clk.fit(X, Y, memoize_targets=memoize_targets)
        pred = clk.transform(X)
        diff = numpy.trace(pred.T @ memoize_targets[0]) / X.shape[0]
        self.assertGreater(diff, 0)


if __name__ == "__main__":
    unittest.main()
