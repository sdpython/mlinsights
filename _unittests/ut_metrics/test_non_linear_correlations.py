# -*- coding: utf-8 -*-
"""
@brief      test log(time=12s)
"""
import unittest
import pandas
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from mlinsights.metrics import non_linear_correlations


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
        self.assertGreater(cor.values.min(), 0)

    def test_non_linear_correlations_array(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        iris = datasets.load_iris()
        X = iris.data[:, :4]
        df = pandas.DataFrame(X).values
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
        self.assertGreater(cor.values.min(), 0)

    def test_non_linear_correlations_df_minmax(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        iris = datasets.load_iris()
        X = iris.data[:, :4]
        df = pandas.DataFrame(X)
        df.columns = ["X1", "X2", "X3", "X4"]
        cor, mini, maxi = non_linear_correlations(
            df, LinearRegression(fit_intercept=False), minmax=True)
        self.assertEqual(cor.shape, (4, 4))
        self.assertEqual(list(cor.columns), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(cor.index), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(cor.iloc[i, i]
                              for i in range(0, 4)), [1, 1, 1, 1])
        self.assertEqual(list(mini.iloc[i, i]
                              for i in range(0, 4)), [1, 1, 1, 1])
        self.assertEqual(list(maxi.iloc[i, i]
                              for i in range(0, 4)), [1, 1, 1, 1])
        self.assertGreater(cor.values.min(), 0)
        self.assertEqual(list(mini.columns), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(mini.index), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(maxi.columns), ["X1", "X2", "X3", "X4"])
        self.assertEqual(list(maxi.index), ["X1", "X2", "X3", "X4"])
        self.assertEqual(mini.shape, (4, 4))
        self.assertLesser(mini.values.min(), cor.values.min())
        self.assertEqual(maxi.shape, (4, 4))
        self.assertGreater(maxi.values.max(), cor.values.max())

    def test_non_linear_correlations_array_minmax(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        iris = datasets.load_iris()
        X = iris.data[:, :4]
        df = pandas.DataFrame(X).values
        cor, mini, maxi = non_linear_correlations(
            df, LinearRegression(fit_intercept=False), minmax=True)
        self.assertEqual(cor.shape, (4, 4))
        self.assertEqual(list(cor[i, i] for i in range(0, 4)), [1, 1, 1, 1])
        self.assertEqual(list(mini[i, i] for i in range(0, 4)), [1, 1, 1, 1])
        self.assertEqual(list(maxi[i, i] for i in range(0, 4)), [1, 1, 1, 1])
        self.assertGreater(cor.min(), 0)
        self.assertEqual(mini.shape, (4, 4))
        self.assertLesser(mini.min(), cor.min())
        self.assertEqual(maxi.shape, (4, 4))
        self.assertGreater(maxi.max(), cor.max())
        self.assertTrue(mini.min() != mini.max())


if __name__ == "__main__":
    unittest.main()
