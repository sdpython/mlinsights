# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from numpy.random import random
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv
from mlinsights.mlmodel.piecewise_estimator import PiecewiseRegressor


class TestPiecewiseRegressor(ExtTestCase):

    def test_piecewise_regressor_no_intercept(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.35], [0.2, 0.36]])
        Y = numpy.array([1., 1.1, 1.15, 1.2])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseRegressor()
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
        self.assertGreater(clq.n_estimators_, 1)

    @ignore_warnings(UserWarning)
    def test_piecewise_regressor_no_intercept_bins(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.35], [0.2, 0.36]])
        Y = numpy.array([1., 1.1, 1.15, 1.2])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseRegressor(binner="bins")
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
        self.assertIn(paths.shape, ((4, 7), (4, 8), (4, 9), (4, 10)))
        self.assertNotEqual(pred2.min(), pred2.max())

    def test_piecewise_regressor_intercept_weights3(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([1., 1.1, 1.2])
        W = numpy.array([1., 1., 1.])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y, W)
        clq = PiecewiseRegressor(verbose=False)
        clq.fit(X, Y, W)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertNotEqual(pred2.min(), pred2.max())
        self.assertEqual(pred1, pred2)

    def test_piecewise_regressor_intercept_weights6(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.3],
                         [0.1, 0.2], [0.2, 0.3], [0.3, 0.3]])
        Y = numpy.array([1., 1.1, 1.2, 1., 1.1, 1.2])
        W = numpy.array([1., 1., 1., 1., 1., 1.])
        clr = LinearRegression(fit_intercept=True)
        clr.fit(X, Y, W)
        clq = PiecewiseRegressor(verbose=False)
        clq.fit(X, Y, W)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertNotEqual(pred2.min(), pred2.max())
        self.assertEqual(pred1, pred2)

    def test_piecewise_regressor_diff(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        clr = LinearRegression()
        clr.fit(X, Y)
        clq = PiecewiseRegressor(verbose=False)
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

    def test_piecewise_regressor_pandas(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [0.2, 0.3]]))
        Y = numpy.array([1., 1.1])
        clr = LinearRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseRegressor()
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertEqual(pred1, pred2)

    def test_piecewise_regressor_list(self):
        X = [[0.1, 0.2], [0.2, 0.3]]
        Y = numpy.array([1., 1.1])
        clq = PiecewiseRegressor()
        self.assertRaise(lambda: clq.fit(X, Y), TypeError)

    def test_piecewise_regressor_pickle(self):
        X = random(100)
        eps1 = (random(90) - 0.5) * 0.1
        eps2 = random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))  # pylint: disable=E1101
        Y = X.ravel() * 3.4 + 5.6 + eps
        test_sklearn_pickle(lambda: LinearRegression(), X, Y)
        test_sklearn_pickle(lambda: PiecewiseRegressor(), X, Y)

    def test_piecewise_regressor_clone(self):
        test_sklearn_clone(lambda: PiecewiseRegressor(verbose=True))

    def test_piecewise_regressor_grid_search(self):
        X = random(100)
        eps1 = (random(90) - 0.5) * 0.1
        eps2 = random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))  # pylint: disable=E1101
        Y = X.ravel() * 3.4 + 5.6 + eps
        self.assertRaise(lambda: test_sklearn_grid_search_cv(
            lambda: PiecewiseRegressor(), X, Y), ValueError)
        res = test_sklearn_grid_search_cv(lambda: PiecewiseRegressor(),
                                          X, Y, binner__max_depth=[2, 3])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 1)

    def test_piecewise_regressor_issue(self):
        X, y = make_regression(10000, n_features=1, n_informative=1,  # pylint: disable=W0632
                               n_targets=1)
        y = y.reshape((-1, 1))
        model = PiecewiseRegressor(
            binner=DecisionTreeRegressor(min_samples_leaf=300))
        model.fit(X, y)
        vvc = model.predict(X)
        self.assertEqual(vvc.shape, (X.shape[0], ))

    def test_piecewise_regressor_raise(self):
        X, y = make_regression(10000, n_features=2, n_informative=2,  # pylint: disable=W0632
                               n_targets=2)
        model = PiecewiseRegressor(
            binner=DecisionTreeRegressor(min_samples_leaf=300))
        self.assertRaise(lambda: model.fit(X, y), RuntimeError)


if __name__ == "__main__":
    unittest.main()
