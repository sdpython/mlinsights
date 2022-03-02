# -*- coding: utf-8 -*-
"""
@brief      test log(time=12s)
"""
import unittest
import pandas
import numpy
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pyquickhelper.pycode import ExtTestCase
from mlinsights.metrics import r2_score_comparable


class TestScoringMetrics(ExtTestCase):

    def test_r2_score_comparable(self):
        iris = datasets.load_iris()
        X = iris.data[:, :4]
        y = iris.target + 1
        df = pandas.DataFrame(X)
        df.columns = ["X1", "X2", "X3", "X4"]
        model1 = LinearRegression().fit(X, y)
        model2 = LinearRegression().fit(X, numpy.log(y))
        r2a = r2_score(y, model1.predict(X))
        r2b = r2_score(numpy.log(y), model2.predict(X))
        r2c = r2_score_comparable(y, model2.predict(X), tr='log')
        r2d = r2_score_comparable(y, model2.predict(X), inv_tr='exp')
        self.assertEqual(r2b, r2c)
        self.assertGreater(r2c, r2a)
        self.assertLesser(r2a, r2d)
        r2e = r2_score_comparable(y, model2.predict(X), inv_tr='exp', tr='exp')
        self.assertLesser(r2e, 0)

    def test_r2_score_comparable_exception(self):
        iris = datasets.load_iris()
        y = iris.target + 1
        self.assertRaise(lambda: r2_score_comparable(y, y), ValueError)
        self.assertRaise(
            lambda: r2_score_comparable(y, y, tr="log2"),
            TypeError)
        self.assertRaise(
            lambda: r2_score_comparable(y, y, inv_tr="log2"),
            TypeError)


if __name__ == "__main__":
    unittest.main()
