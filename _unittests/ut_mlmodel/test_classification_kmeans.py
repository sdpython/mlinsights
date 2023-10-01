# -*- coding: utf-8 -*-
import unittest
import numpy
from numpy.random import RandomState
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel import (
    ClassifierAfterKMeans,
    run_test_sklearn_pickle,
    run_test_sklearn_clone,
    run_test_sklearn_grid_search_cv,
)


class TestClassifierAfterKMeans(ExtTestCase):
    @ignore_warnings(category=ConvergenceWarning)
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

    @ignore_warnings(category=ConvergenceWarning)
    def test_classification_kmeans_intercept_weights(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr = ClassifierAfterKMeans()
        clr.fit(X, y, sample_weight=numpy.ones((X.shape[0],)))
        acc = clr.score(X, y)
        self.assertGreater(acc, 0)

    @ignore_warnings(category=ConvergenceWarning)
    def test_classification_kmeans_pickle(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        run_test_sklearn_pickle(lambda: ClassifierAfterKMeans(), X, y)

    def test_classification_kmeans_clone(self):
        self.maxDiff = None
        run_test_sklearn_clone(lambda: ClassifierAfterKMeans())

    @ignore_warnings(category=ConvergenceWarning)
    def test_classification_kmeans_grid_search(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        self.assertRaise(
            lambda: run_test_sklearn_grid_search_cv(
                lambda: ClassifierAfterKMeans(), X, y
            ),
            ValueError,
        )
        res = run_test_sklearn_grid_search_cv(
            lambda: ClassifierAfterKMeans(), X, y, c_n_clusters=[2, 3]
        )
        self.assertIn("model", res)
        self.assertIn("score", res)
        self.assertGreater(res["score"], 0)
        self.assertLesser(res["score"], 1)

    @ignore_warnings(category=ConvergenceWarning)
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

    @ignore_warnings(category=ConvergenceWarning)
    def test_issue(self):
        X, labels_true = datasets.make_blobs(n_samples=750, centers=6, cluster_std=0.4)[
            :2
        ]
        labels_true = labels_true % 3
        clcl = ClassifierAfterKMeans(e_max_iter=1000)
        clcl.fit(X, labels_true)
        r = repr(clcl)
        self.assertIn("ClassifierAfterKMeans(", r)
        self.assertIn("c_init='k-means++'", r)


if __name__ == "__main__":
    unittest.main()
