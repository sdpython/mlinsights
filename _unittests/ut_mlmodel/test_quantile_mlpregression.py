# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import numpy
import pandas
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
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

from src.mlinsights.mlmodel import QuantileMLPRegressor
from src.mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv


class TestQuantileMLPRegression(ExtTestCase):

    def test_quantile_regression_diff(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        clr = MLPRegressor(hidden_layer_sizes=(3,))
        clr.fit(X, Y)
        clq = QuantileMLPRegressor(hidden_layer_sizes=(3,))
        clq.fit(X, Y)
        self.assertGreater(clr.n_iter_, 10)
        self.assertGreater(clq.n_iter_, 10)
        err1 = mean_absolute_error(Y, clr.predict(X))
        err2 = mean_absolute_error(Y, clq.predict(X))
        self.assertLesser(err1, 5)
        self.assertLesser(err2, 5)

    def test_quantile_regression_pandas(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [0.2, 0.3]]))
        Y = numpy.array([1., 1.1])
        clr = MLPRegressor(hidden_layer_sizes=(3,))
        clr.fit(X, Y)
        clq = QuantileMLPRegressor(hidden_layer_sizes=(3,))
        clq.fit(X, Y)
        self.assertGreater(clr.n_iter_, 10)
        self.assertGreater(clq.n_iter_, 10)
        err1 = mean_absolute_error(Y, clr.predict(X))
        err2 = mean_absolute_error(Y, clq.predict(X))
        self.assertLesser(err1, 3)
        self.assertLesser(err2, 3)

    def test_quantile_regression_pickle(self):
        X = numpy.random.random(100)
        eps1 = (numpy.random.random(90) - 0.5) * 0.1
        eps2 = numpy.random.random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))
        Y = X.ravel() * 3.4 + 5.6 + eps
        test_sklearn_pickle(lambda: MLPRegressor(
            hidden_layer_sizes=(3,)), X, Y)
        test_sklearn_pickle(lambda: QuantileMLPRegressor(
            hidden_layer_sizes=(3,)), X, Y)

    def test_quantile_regression_clone(self):
        test_sklearn_clone(lambda: QuantileMLPRegressor())

    def test_quantile_regression_grid_search(self):
        X = numpy.random.random(100)
        eps1 = (numpy.random.random(90) - 0.5) * 0.1
        eps2 = numpy.random.random(10) * 2
        eps = numpy.hstack([eps1, eps2])
        X = X.reshape((100, 1))
        Y = X.ravel() * 3.4 + 5.6 + eps
        self.assertRaise(lambda: test_sklearn_grid_search_cv(
            lambda: QuantileMLPRegressor(hidden_layer_sizes=(3,)), X, Y), ValueError)
        res = test_sklearn_grid_search_cv(lambda: QuantileMLPRegressor(hidden_layer_sizes=(3,)),
                                          X, Y, learning_rate_init=[0.001, 0.0001])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 10)


if __name__ == "__main__":
    unittest.main()
