# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
import numpy
from numpy.random import RandomState
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import TSNE
from pyquickhelper.pycode import (
    ExtTestCase, skipif_circleci, ignore_warnings)
from mlinsights.mlmodel import PredictableTSNE
from mlinsights.mlmodel import run_test_sklearn_pickle, run_test_sklearn_clone


class TestPredictableTSNE(ExtTestCase):

    @ignore_warnings(ConvergenceWarning)
    def test_predictable_tsne(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        clr = PredictableTSNE(keep_tsne_outputs=True)
        clr.fit(X, y)
        pred = clr.transform(X)
        self.assertIsInstance(clr.estimator_, MLPRegressor)
        self.assertGreater(clr.loss_, 0)
        self.assertNotEmpty(pred)

    @skipif_circleci('stuck')
    @ignore_warnings(ConvergenceWarning)
    def test_predictable_tsne_knn(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        clr = PredictableTSNE(estimator=KNeighborsRegressor(),
                              keep_tsne_outputs=True)
        clr.fit(X, y)
        pred = clr.transform(X)
        self.assertTrue(hasattr(clr, "tsne_outputs_"))
        self.assertIsInstance(clr.estimator_, KNeighborsRegressor)
        self.assertEqual(pred.shape, (X.shape[0], 2))

    @ignore_warnings(ConvergenceWarning)
    def test_predictable_tsne_intercept_weights(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        clr = PredictableTSNE(keep_tsne_outputs=True)
        clr.fit(X, y, sample_weight=numpy.ones((X.shape[0],)))
        acc = clr.transform(X)
        self.assertGreater(clr.loss_, 0)
        self.assertEqual(acc.shape, (X.shape[0], 2))

    @ignore_warnings(ConvergenceWarning)
    def test_predictable_tsne_pickle(self):
        iris = datasets.load_iris()
        X, y = iris.data[:20], iris.target[:20]
        run_test_sklearn_pickle(lambda: PredictableTSNE(), X, y)

    @ignore_warnings(ConvergenceWarning)
    def test_predictable_tsne_clone(self):
        self.maxDiff = None
        run_test_sklearn_clone(lambda: PredictableTSNE())

    @ignore_warnings(ConvergenceWarning)
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
        clk = PredictableTSNE(transformer=TSNE(n_components=2),
                              normalizer=StandardScaler(with_mean=False),
                              keep_tsne_outputs=True)
        clk.fit(X, Y)
        pred = clk.transform(X)
        self.assertGreater(clk.loss_, 0)
        self.assertEqual(pred.shape, (X.shape[0], 2))


if __name__ == "__main__":
    unittest.main()
