# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from numpy.random import random
import pandas
from sklearn import __version__ as sklver
from sklearn.linear_model import LinearRegression
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper import compare_module_version
from mlinsights.mlmodel import QuantileLinearRegression
from mlinsights.mlmodel import (
    run_test_sklearn_pickle,
    run_test_sklearn_clone,
    run_test_sklearn_grid_search_cv)
from mlinsights.mlmodel.quantile_mlpregressor import float_sign


class TestQuantileRegression(ExtTestCase):

    def test_sklver(self):
        self.assertTrue(compare_module_version(sklver, "0.22") >= 0)

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

    @unittest.skipIf(
        compare_module_version(sklver, "0.24") == -1,
        reason="positive was introduce in 0.24")
    def test_quantile_regression_no_intercept_positive(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3]])
        Y = numpy.array([1., 1.1])
        clr = LinearRegression(fit_intercept=False, positive=True)
        clr.fit(X, Y)
        clq = QuantileLinearRegression(fit_intercept=False, positive=True)
        clq.fit(X, Y)
        self.assertEqual(clr.intercept_, 0)
        self.assertEqual(clq.intercept_, 0)
        self.assertGreater(clr.coef_.min(), 0)
        self.assertGreater(clq.coef_.min(), 0)
        self.assertEqualArray(clr.intercept_, clq.intercept_)
        self.assertEqualArray(clr.coef_[0], clq.coef_[0])
        self.assertGreater(clr.coef_[1:].min(), 3)
        self.assertGreater(clq.coef_[1:].min(), 3)

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
        self.assertEqualArray(clr.coef_, clq.coef_, atol=1e-10)

    @unittest.skipIf(
        compare_module_version(sklver, "0.24") == -1,
        reason="positive was introduce in 0.24")
    def test_quantile_regression_intercept_positive(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([1., 1.1, 1.2])
        clr = LinearRegression(fit_intercept=True, positive=True)
        clr.fit(X, Y)
        clq = QuantileLinearRegression(
            verbose=False, fit_intercept=True, positive=True)
        clq.fit(X, Y)
        self.assertNotEqual(clr.intercept_, 0)
        self.assertNotEqual(clq.intercept_, 0)
        self.assertEqualArray(clr.intercept_, clq.intercept_)
        self.assertEqualArray(clr.coef_, clq.coef_, atol=1e-10)
        self.assertGreater(clr.coef_.min(), 0)
        self.assertGreater(clq.coef_.min(), 0)

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
        self.assertEqualArray(clr.coef_, clq.coef_, atol=1e-10)

    def test_quantile_regression_diff(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
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
        X = random(1000)
        eps1 = (random(900) - 0.5) * 0.1
        eps2 = random(100) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((1000, 1))  # pylint: disable=E1101
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
        X = random(100)
        eps1 = (random(90) - 0.5) * 0.1
        eps2 = random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))  # pylint: disable=E1101
        Y = X.ravel() * 3.4 + 5.6 + eps
        run_test_sklearn_pickle(lambda: LinearRegression(), X, Y)
        run_test_sklearn_pickle(lambda: QuantileLinearRegression(), X, Y)

    def test_quantile_regression_clone(self):
        run_test_sklearn_clone(lambda: QuantileLinearRegression(delta=0.001))

    def test_quantile_regression_grid_search(self):
        X = random(100)
        eps1 = (random(90) - 0.5) * 0.1
        eps2 = random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))  # pylint: disable=E1101
        Y = X.ravel() * 3.4 + 5.6 + eps
        self.assertRaise(lambda: run_test_sklearn_grid_search_cv(
            lambda: QuantileLinearRegression(), X, Y),
            (ValueError, TypeError))
        res = run_test_sklearn_grid_search_cv(lambda: QuantileLinearRegression(),
                                              X, Y, delta=[0.1, 0.001])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 1)

    def test_quantile_regression_diff_quantile(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
        Y = numpy.array([1., 1.11, 1.21, 10, 1.29, 1.39])
        clqs = []
        scores = []
        for q in [0.25, 0.4999, 0.5, 0.5001, 0.75]:
            clq = QuantileLinearRegression(
                verbose=False, fit_intercept=True, quantile=q)
            clq.fit(X, Y)
            clqs.append(clq)
            sc = clq.score(X, Y)
            scores.append(sc)
            self.assertGreater(sc, 0)

        self.assertLesser(abs(clqs[1].intercept_ - clqs[2].intercept_), 0.01)
        self.assertLesser(abs(clqs[2].intercept_ - clqs[3].intercept_), 0.01)
        self.assertLesser(abs(clqs[1].coef_[0] - clqs[2].coef_[0]), 0.01)
        self.assertLesser(abs(clqs[2].coef_[0] - clqs[3].coef_[0]), 0.01)

        self.assertGreater(abs(clqs[0].intercept_ - clqs[1].intercept_), 0.01)
        # self.assertGreater(abs(clqs[3].intercept_ - clqs[4].intercept_), 0.01)
        self.assertGreater(abs(clqs[0].coef_[0] - clqs[1].coef_[0]), 0.05)
        # self.assertGreater(abs(clqs[3].coef_[0] - clqs[4].coef_[0]), 0.05)

        self.assertLesser(abs(scores[1] - scores[2]), 0.01)
        self.assertLesser(abs(scores[2] - scores[3]), 0.01)

    def test_quantile_regression_quantile_check(self):
        n = 100
        X = numpy.arange(n) / n
        Y = X + X * X / n
        X = X.reshape((n, 1))
        for q in [0.1, 0.5, 0.9]:
            clq = QuantileLinearRegression(
                verbose=False, fit_intercept=True, quantile=q, max_iter=10)
            clq.fit(X, Y)
            y = clq.predict(X)
            diff = y - Y
            sign = numpy.sign(diff)  # pylint: disable=E1111
            pos = (sign > 0).sum()  # pylint: disable=W0143
            neg = (sign < 0).sum()  # pylint: disable=W0143
            if q < 0.5:
                self.assertGreater(neg, pos * 4)
            if q > 0.5:
                self.assertLesser(neg * 7, pos)

    def test_float_sign(self):
        self.assertEqual(float_sign(-1), -1)
        self.assertEqual(float_sign(1), 1)
        self.assertEqual(float_sign(1e-16), 0)

    def test_quantile_regression_intercept_D2(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([[1., 0.], [1.1, 0.1], [1.2, 0.19]])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y)
        clq = QuantileLinearRegression(verbose=False, fit_intercept=True)
        self.assertRaise(lambda: clq.fit(X, Y), ValueError)


if __name__ == "__main__":
    unittest.main()
