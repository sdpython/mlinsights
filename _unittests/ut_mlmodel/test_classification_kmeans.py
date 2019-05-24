# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from numpy.random import RandomState
from sklearn import datasets
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel import ClassifierAfterKMeans
from mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv


class TestClassifierAfterKMeans(ExtTestCase):

    def test_classification_kmeans(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr = ClassifierAfterKMeans()
        clr.fit(X, y)
        acc = clr.score(X, y)
        self.assertGreater(acc, 0)
        prob = clr.predict_proba(X)
        self.assertEqual(prob.shape[1], 3)
        dec = clr.decision_function(X)
        self.assertEqual(prob.shape, dec.shape)

    def test_classification_kmeans_intercept_weights(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr = ClassifierAfterKMeans()
        clr.fit(X, y, sample_weight=numpy.ones((X.shape[0],)))
        acc = clr.score(X, y)
        self.assertGreater(acc, 0)

    def test_classification_kmeans_pickle(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        test_sklearn_pickle(lambda: ClassifierAfterKMeans(), X, y)

    def test_classification_kmeans_clone(self):
        self.maxDiff = None
        test_sklearn_clone(lambda: ClassifierAfterKMeans())

    def test_classification_kmeans_grid_search(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        self.assertRaise(lambda: test_sklearn_grid_search_cv(
            lambda: ClassifierAfterKMeans(), X, y), ValueError)
        res = test_sklearn_grid_search_cv(lambda: ClassifierAfterKMeans(), X, y,
                                          c_n_clusters=[2, 3])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 1)

    def test_classification_kmeans_relevance(self):
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
        clk = ClassifierAfterKMeans(c_n_clusters=6, c_random_state=state)
        clk.fit(X, Y)
        score = clk.score(X, Y)
        self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()
