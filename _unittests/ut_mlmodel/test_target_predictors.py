# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel.sklearn_transform_inv_fct import FunctionReciprocalTransformer
from mlinsights.mlmodel import TransformedTargetClassifier2, TransformedTargetRegressor2


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

    def test_target_classifier(self):
        tt = TransformedTargetClassifier2(classifier=None, transformer='rnd')
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.array([0, 0, 1, 1], dtype=int)
        tt.fit(X, y)
        self.assertIn("TransformedTargetClassifier2", str(tt))
        coef = tt.classifier_.coef_
        self.assertEqual(coef.shape, (1, 1))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4, ))
        sc = tt.score(X, y)
        self.assertEqual(sc, 1.)

    def test_target_classifier_proba(self):
        tt = TransformedTargetClassifier2(classifier=None, transformer='rnd')
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.array([0, 0, 1, 1], dtype=int)
        tt.fit(X, y)
        cl = tt.classes_
        self.assertEqual(cl.shape, tt.classifier_.classes_.shape)
        yp2 = tt.classifier_.predict_proba(tt.transformer_.transform(X, y)[0])
        if tt.transformer_.permutation_[0] == 0:
            self.assertEqualArray(cl, tt.classifier_.classes_)
        else:
            self.assertEqualArray(cl, -(tt.classifier_.classes_ - 1))
            c = yp2.copy()
            yp2[:, 0] = c[:, 1]
            yp2[:, 1] = c[:, 0]
        yp = tt.predict_proba(X)
        self.assertEqual(yp.shape, (4, 2))
        self.assertEqualArray(yp, yp2)

    def test_target_classifier_decision(self):
        tt = TransformedTargetClassifier2(classifier=None, transformer='rnd')
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.array([0, 0, 1, 1], dtype=int)
        tt.fit(X, y)
        self.assertRaise(lambda: tt.decision_function(X), RuntimeError)

    def test_target_regressor_err(self):
        tt = TransformedTargetRegressor2(regressor=None, transformer=None)
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        self.assertRaise(lambda: tt.fit(X, y), TypeError)

    def test_target_classifier_err(self):
        tt = TransformedTargetClassifier2(classifier=None, transformer=None)
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

    def test_target_classifier_any(self):
        trans = FunctionReciprocalTransformer('log')
        tt = TransformedTargetClassifier2(classifier=None, transformer=trans)
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        tt.fit(X, y)
        self.assertIn("TransformedTargetClassifier2", str(tt))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4, ))


if __name__ == "__main__":
    unittest.main()
