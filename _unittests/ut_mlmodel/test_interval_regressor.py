# -*- coding: utf-8 -*-
import unittest
import numpy
from sklearn.linear_model import LinearRegression
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel import IntervalRegressor


class TestIntervalRegressor(ExtTestCase):
    def test_interval_regressor(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.35], [0.2, 0.36]])
        Y = numpy.array([1.0, 1.1, 1.15, 1.2])
        clr = IntervalRegressor(n_estimators=2, estimator=LinearRegression())
        clr.fit(X, Y)
        pred = clr.predict(X)
        preds = clr.predict_sorted(X)
        self.assertEqual(pred.shape, (4,))
        self.assertEqual(preds.shape, (4, 2))
        rnd = preds[:, 0] <= preds[:, 1]
        nb = rnd.sum()
        self.assertEqual(nb, 4)
        self.assertEqual(clr.n_estimators_, 2)


if __name__ == "__main__":
    unittest.main()
