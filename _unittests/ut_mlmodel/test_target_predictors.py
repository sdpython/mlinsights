# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel.sklearn_transform_inv_fct import FunctionReciprocalTransformer
from mlinsights.mlmodel import TransformedTargetRegressor2


class TestTargetPredictors(ExtTestCase):

    def test_target_regressor(self):
        tt = TransformedTargetRegressor2(regressor=None, transformer='log')
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        tt.fit(X, y)
        self.assertIn("TransformedTargetRegressor2", str(tt))
        coef = tt.regressor_.coef_
        self.assertEqualArray(coef, numpy.array([2], dtype=float))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4, ))
        sc = tt.score(X, y)
        self.assertEqual(sc, 1.)

    def test_target_regressor_err(self):
        tt = TransformedTargetRegressor2(regressor=None, transformer=None)
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        self.assertRaise(lambda: tt.fit(X, y), TypeError)

    def test_target_regressor_any(self):
        trans = FunctionReciprocalTransformer('log')
        tt = TransformedTargetRegressor2(regressor=None, transformer=trans)
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        tt.fit(X, y)
        self.assertIn("TransformedTargetRegressor2", str(tt))
        coef = tt.regressor_.coef_
        self.assertEqualArray(coef, numpy.array([2], dtype=float))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4, ))
        sc = tt.score(X, y)
        self.assertEqual(sc, 1.)


if __name__ == "__main__":
    unittest.main()
