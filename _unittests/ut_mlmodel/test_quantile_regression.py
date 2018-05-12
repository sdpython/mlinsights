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

from src.mlinsights.mlmodel import QuantileLinearRegression
from src.mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv


class TestQuantileRegression(ExtTestCase):

    def test_quantile_regression_no_intercept(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3]])
        Y = numpy.array([1., 1.1])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = QuantileLinearRegression(fit_intercept=False)
        clq.fit(X, Y)
        self.assertEqual(clr.intercept_, 0)
        self.assertEqualArray(clr.coef_, clq.coef_)
        self.assertEqual(clq.intercept_, 0)
        self.assertEqualArray(clr.intercept_, clq.intercept_)

    def test_quantile_regression_intercept(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([1., 1.1, 1.2])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y)
        clq = QuantileLinearRegression(verbose=False, fit_intercept=True)
        clq.fit(X, Y)
        self.assertNotEqual(clr.intercept_, 0)
        self.assertNotEqual(clq.intercept_, 0)
        self.assertEqualArray(clr.intercept_, clq.intercept_)
        self.assertEqualArray(clr.coef_, clq.coef_)

    def test_quantile_regression_intercept_weights(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([1., 1.1, 1.2])
        W = numpy.array([1., 1., 1.])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y, W)
        clq = QuantileLinearRegression(verbose=False, fit_intercept=True)
        clq.fit(X, Y, W)
        self.assertNotEqual(clr.intercept_, 0)
        self.assertNotEqual(clq.intercept_, 0)
        self.assertEqualArray(clr.intercept_, clq.intercept_)
        self.assertEqualArray(clr.coef_, clq.coef_)

    def test_quantile_regression_diff(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4]])
        Y = numpy.array([1., 1.1, 1.2, 10])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y)
        clq = QuantileLinearRegression(verbose=False, fit_intercept=True)
        clq.fit(X, Y)
        self.assertNotEqual(clr.intercept_, 0)
        self.assertNotEqual(clq.intercept_, 0)
        self.assertNotEqualArray(clr.coef_, clq.coef_)
        self.assertNotEqualArray(clr.intercept_, clq.intercept_)
        self.assertLesser(clq.n_iter_, 10)

    def test_quantile_regression_pandas(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [0.2, 0.3]]))
        Y = numpy.array([1., 1.1])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = QuantileLinearRegression(fit_intercept=False)
        clq.fit(X, Y)
        self.assertEqual(clr.intercept_, 0)
        self.assertEqualArray(clr.coef_, clq.coef_)
        self.assertEqual(clq.intercept_, 0)
        self.assertEqualArray(clr.intercept_, clq.intercept_)

    def test_quantile_regression_list(self):
        X = [[0.1, 0.2], [0.2, 0.3]]
        Y = numpy.array([1., 1.1])
        clq = QuantileLinearRegression(fit_intercept=False)
        self.assertRaise(lambda: clq.fit(X, Y), TypeError)

    def test_quantile_regression_list2(self):
        X = numpy.random.random(1000)
        eps1 = (numpy.random.random(900) - 0.5) * 0.1
        eps2 = numpy.random.random(100) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((1000, 1))
        Y = X * 3.4 + 5.6 + eps

        clq = QuantileLinearRegression(verbose=False, fit_intercept=True)
        self.assertRaise(lambda: clq.fit(X, Y), ValueError)

        Y = X.ravel() * 3.4 + 5.6 + eps

        clq = QuantileLinearRegression(verbose=False, fit_intercept=True)
        clq.fit(X, Y)

        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y)

        self.assertNotEqual(clr.intercept_, 0)
        self.assertNotEqual(clq.intercept_, 0)
        self.assertNotEqualArray(clr.coef_, clq.coef_)
        self.assertNotEqualArray(clr.intercept_, clq.intercept_)
        self.assertLesser(clq.n_iter_, 10)

        pr = clr.predict(X)
        pq = clq.predict(X)
        self.assertEqual(pr.shape, pq.shape)

    def test_quantile_regression_pickle(self):
        X = numpy.random.random(100)
        eps1 = (numpy.random.random(90) - 0.5) * 0.1
        eps2 = numpy.random.random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))
        Y = X.ravel() * 3.4 + 5.6 + eps
        test_sklearn_pickle(lambda: LinearRegression(), X, Y)
        test_sklearn_pickle(lambda: QuantileLinearRegression(), X, Y)

    def test_quantile_regression_clone(self):
        test_sklearn_clone(lambda: QuantileLinearRegression(delta=0.001))

    def test_quantile_regression_grid_search(self):
        X = numpy.random.random(100)
        eps1 = (numpy.random.random(90) - 0.5) * 0.1
        eps2 = numpy.random.random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))
        Y = X.ravel() * 3.4 + 5.6 + eps
        self.assertRaise(lambda: test_sklearn_grid_search_cv(
            lambda: QuantileLinearRegression(), X, Y), ValueError)
        res = test_sklearn_grid_search_cv(lambda: QuantileLinearRegression(),
                                          X, Y, delta=[0.1, 0.001])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 1)


if __name__ == "__main__":
    unittest.main()
