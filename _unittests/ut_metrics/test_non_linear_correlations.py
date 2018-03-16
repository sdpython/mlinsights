#-*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import pandas
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    import pyquickhelper as skip_


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

from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from src.mlinsights.metrics import non_linear_correlations


class TestNonLinearCorrelations(ExtTestCase):

    def test_non_linear_correlations_df(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        iris = datasets.load_iris()
        X = iris.data[:, :4]
        df = pandas.DataFrame(X)
        df.columns = ["X1", "X2", "X3", "X4"]
        cor = non_linear_correlations(
            df, LinearRegression(fit_intercept=False))
        self.assertEqual(cor.shape, (4, 4))
        self.assertEqual(list(cor.columns), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(cor.index), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(cor.iloc[i, i]
                              for i in range(0, 4)), [1, 1, 1, 1])
        self.assertGreater(cor.as_matrix().min(), 0)

    def test_non_linear_correlations_array(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        iris = datasets.load_iris()
        X = iris.data[:, :4]
        df = pandas.DataFrame(X).as_matrix()
        cor = non_linear_correlations(
            df, LinearRegression(fit_intercept=False))
        self.assertEqual(cor.shape, (4, 4))
        self.assertEqual(list(cor[i, i] for i in range(0, 4)), [1, 1, 1, 1])
        self.assertGreater(cor.min(), 0)

    def test_non_linear_correlations_df_tree(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        iris = datasets.load_iris()
        X = iris.data[:, :4]
        df = pandas.DataFrame(X)
        df.columns = ["X1", "X2", "X3", "X4"]
        cor = non_linear_correlations(df, RandomForestRegressor())
        self.assertEqual(cor.shape, (4, 4))
        self.assertEqual(list(cor.columns), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(cor.index), ["X1", "X2", "X3", "X4"])
        self.assertGreater(max(cor.iloc[i, i] for i in range(0, 4)), 0.98)
        self.assertGreater(cor.as_matrix().min(), 0)


if __name__ == "__main__":
    unittest.main()
