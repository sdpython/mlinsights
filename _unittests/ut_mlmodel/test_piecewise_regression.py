# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import numpy
import pandas
from sklearn.linear_model import LinearRegression
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

from src.mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv
from src.mlinsights.mlmodel.piecewise_linear_regression import PiecewiseLinearRegression


class TestPiecewiseRegression(ExtTestCase):

    def test_piecewise_regression_no_intercept(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.35], [0.2, 0.36]])
        Y = numpy.array([1., 1.1, 1.15, 1.2])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseLinearRegression()
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertEqual(pred1.shape, (4,))
        self.assertEqual(pred2.shape, (4,))
        sc1 = clr.score(X, Y)
        sc2 = clq.score(X, Y)
        sc3 = clq.binner_.score(X, Y)
        self.assertIsInstance(sc1, float)
        self.assertIsInstance(sc2, float)
        self.assertIsInstance(sc3, float)
        paths = clq.binner_.decision_path(X)
        s = paths.sum()
        self.assertEqual(s, 8)
        self.assertNotEqual(pred2.min(), pred2.max())

    def test_piecewise_regression_no_intercept_bins(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.35], [0.2, 0.36]])
        Y = numpy.array([1., 1.1, 1.15, 1.2])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseLinearRegression(binner="bins")
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertEqual(pred1.shape, (4,))
        self.assertEqual(pred2.shape, (4,))
        sc1 = clr.score(X, Y)
        sc2 = clq.score(X, Y)
        self.assertIsInstance(sc1, float)
        self.assertIsInstance(sc2, float)
        paths = clq.binner_.transform(X)
        self.assertEqual(paths.shape, (4, 10))
        self.assertNotEqual(pred2.min(), pred2.max())

    def test_piecewise_regression_intercept_weights3(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([1., 1.1, 1.2])
        W = numpy.array([1., 1., 1.])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y, W)
        clq = PiecewiseLinearRegression(verbose=False)
        clq.fit(X, Y, W)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertNotEqual(pred2.min(), pred2.max())
        self.assertEqual(pred1, pred2)

    def test_piecewise_regression_intercept_weights6(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3],
                         [0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([1., 1.1, 1.2, 1., 1.1, 1.2])
        W = numpy.array([1., 1., 1., 1., 1., 1.])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y, W)
        clq = PiecewiseLinearRegression(verbose=False)
        clq.fit(X, Y, W)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertNotEqual(pred2.min(), pred2.max())
        self.assertEqual(pred1, pred2)

    def test_piecewise_regression_diff(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        clr = LinearRegression()
        clr.fit(X, Y)
        clq = PiecewiseLinearRegression(verbose=False)
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        self.assertNotEmpty(pred1)
        pred2 = clq.predict(X)
        self.assertEqual(len(clq.estimators_), 2)
        p1 = clq.estimators_[0].predict(X[:3, :])
        p2 = clq.estimators_[1].predict(X[3:, :])
        self.assertEqual(pred2[:3], p1)
        self.assertEqual(pred2[-2:], p2)
        sc = clq.score(X, Y)
        self.assertEqual(sc, 1)

    def test_piecewise_regression_pandas(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [0.2, 0.3]]))
        Y = numpy.array([1., 1.1])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseLinearRegression()
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertEqual(pred1, pred2)

    def test_piecewise_regression_list(self):
        X = [[0.1, 0.2], [0.2, 0.3]]
        Y = numpy.array([1., 1.1])
        clq = PiecewiseLinearRegression()
        self.assertRaise(lambda: clq.fit(X, Y), TypeError)

    def test_piecewise_regression_pickle(self):
        X = numpy.random.random(100)
        eps1 = (numpy.random.random(90) - 0.5) * 0.1
        eps2 = numpy.random.random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))
        Y = X.ravel() * 3.4 + 5.6 + eps
        test_sklearn_pickle(lambda: LinearRegression(), X, Y)
        test_sklearn_pickle(lambda: PiecewiseLinearRegression(), X, Y)

    def test_piecewise_regression_clone(self):
        test_sklearn_clone(lambda: PiecewiseLinearRegression(verbose=True))

    def test_piecewise_regression_grid_search(self):
        X = numpy.random.random(100)
        eps1 = (numpy.random.random(90) - 0.5) * 0.1
        eps2 = numpy.random.random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))
        Y = X.ravel() * 3.4 + 5.6 + eps
        self.assertRaise(lambda: test_sklearn_grid_search_cv(
            lambda: PiecewiseLinearRegression(), X, Y), ValueError)
        res = test_sklearn_grid_search_cv(lambda: PiecewiseLinearRegression(),
                                          X, Y, binner__max_depth=[2, 3])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 1)


if __name__ == "__main__":
    unittest.main()
