# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import numpy
import pandas
from sklearn.linear_model import LogisticRegression
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

from src.mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv
from src.mlinsights.mlmodel.piecewise_estimator import PiecewiseClassifier


class TestPiecewiseClassifier(ExtTestCase):

    def test_piecewise_classifier_no_intercept(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]])
        Y = numpy.array([0, 1, 0, 1])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier(random_state=0)
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertEqual(pred1.shape, (4,))
        self.assertEqual(pred2.shape, (4,))
        self.assertEqual(pred1, pred2)
        sc1 = clr.score(X, Y)
        sc2 = clq.score(X, Y)
        sc3 = clq.binner_.score(X, Y)
        self.assertIsInstance(sc1, float)
        self.assertIsInstance(sc2, float)
        self.assertIsInstance(sc3, float)
        paths = clq.binner_.decision_path(X)
        s = paths.sum()
        self.assertEqual(s, 4)
        self.assertNotEqual(pred2.min(), pred2.max())
        self.assertGreater(clq.n_estimators_, 1)

    def test_piecewise_classifier_no_intercept_proba(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]])
        Y = numpy.array([0, 1, 0, 1])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier(random_state=0)
        clq.fit(X, Y)
        pred1 = clr.predict_proba(X)
        pred2 = clq.predict_proba(X)
        self.assertEqual(pred1.shape, (4, 2))
        self.assertEqual(pred2.shape, (4, 2))

    def test_piecewise_classifier_no_intercept_proba_3(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35],
                         [-0.2, -0.36], [-3, 3], [-4, 4]])
        Y = numpy.array([0, 1, 0, 1, 2, 2])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier(random_state=0)
        clq.fit(X, Y)
        pred1 = clr.predict_proba(X)
        pred2 = clq.predict_proba(X)
        self.assertEqual(pred1.shape, (6, 3))
        self.assertEqual(pred2.shape, (6, 3))

    def test_piecewise_classifier_no_intercept_decision(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]])
        Y = numpy.array([0, 1, 0, 1])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier(random_state=0)
        clq.fit(X, Y)
        pred1 = clr.decision_function(X)
        pred2 = clq.decision_function(X)
        self.assertEqual(pred1.shape, (4, ))
        self.assertEqual(pred2.shape, (4, ))

    def test_piecewise_classifier_no_intercept_decision_3(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35],
                         [-0.2, -0.36], [-3, 3], [-4, 4]])
        Y = numpy.array([0, 1, 0, 1, 2, 2])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier(random_state=0)
        clq.fit(X, Y)
        pred1 = clr.decision_function(X)
        pred2 = clq.decision_function(X)
        self.assertEqual(pred1.shape, (6, 3))
        self.assertEqual(pred2.shape, (6, 3))

    def test_piecewise_classifier_no_intercept_predict_3(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35],
                         [-0.2, -0.36], [-3, 3], [-4, 4]])
        Y = numpy.array([0, 1, 0, 1, 2, 2])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier(random_state=0)
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertEqual(pred1.shape, (6, ))
        self.assertEqual(pred2.shape, (6, ))

    def test_piecewise_classifier_no_intercept_bins(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]])
        Y = numpy.array([0, 1, 0, 1])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier(binner="bins")
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
        self.assertEqual(paths.shape, (4, 10))
        self.assertNotEqual(pred2.min(), pred2.max())

    def test_piecewise_classifier_intercept_weights3(self):
        X = numpy.array([[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]])
        Y = numpy.array([0, 1, 0, 1])
        W = numpy.array([1., 1., 1., 1.])
        clr = LogisticRegression(fit_intercept=True)
        clr.fit(X, Y, W)
        clq = PiecewiseClassifier(verbose=False)
        clq.fit(X, Y, W)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertNotEqual(pred2.min(), pred2.max())
        self.assertEqual(pred1, pred2)

    def test_piecewise_classifier_pandas(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [0.2, 0.3]]))
        Y = numpy.array([0, 1])
        clr = LogisticRegression(fit_intercept=False)
        clr.fit(X, Y)
        clq = PiecewiseClassifier()
        clq.fit(X, Y)
        pred1 = clr.predict(X)
        pred2 = clq.predict(X)
        self.assertEqual(pred1, pred2)

    def test_piecewise_classifier_list(self):
        X = [[0.1, 0.2], [0.2, 0.3]]
        Y = numpy.array([0, 1])
        clq = PiecewiseClassifier()
        self.assertRaise(lambda: clq.fit(X, Y), TypeError)

    def test_piecewise_classifier_pickle(self):
        X = numpy.random.random(100)
        Y = X > 0.5  # pylint: disable=W0143
        X = X.reshape((100, 1))  # pylint: disable=E1101
        test_sklearn_pickle(lambda: LogisticRegression(), X, Y)
        test_sklearn_pickle(lambda: PiecewiseClassifier(), X, Y)

    def test_piecewise_classifier_clone(self):
        test_sklearn_clone(lambda: PiecewiseClassifier(verbose=True))

    def test_piecewise_classifier_grid_search(self):
        X = numpy.random.random(100)
        Y = X > 0.5  # pylint: disable=W0143
        X = X.reshape((100, 1))  # pylint: disable=E1101
        self.assertRaise(lambda: test_sklearn_grid_search_cv(
            lambda: PiecewiseClassifier(), X, Y), ValueError)
        res = test_sklearn_grid_search_cv(lambda: PiecewiseClassifier(),
                                          X, Y, binner__max_depth=[2, 3])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 1)


if __name__ == "__main__":
    unittest.main()
