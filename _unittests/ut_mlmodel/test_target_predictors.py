# -*- coding: utf-8 -*-
import unittest
import numpy
from sklearn import __version__ as sklver
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper import compare_module_version
from mlinsights.mlmodel.sklearn_transform_inv_fct import FunctionReciprocalTransformer
from mlinsights.mlmodel import TransformedTargetClassifier2, TransformedTargetRegressor2


class TestTargetPredictors(ExtTestCase):
    def test_target_regressor(self):
        tt = TransformedTargetRegressor2(regressor=None, transformer="log")
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        tt.fit(X, y)
        self.assertIn("TransformedTargetRegressor2", str(tt))
        coef = tt.regressor_.coef_
        self.assertEqualArray(coef, numpy.array([2], dtype=float))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4,))
        sc = tt.score(X, y)
        self.assertLesser(sc, 1.0)

    def test_target_regressor_permute(self):
        tt = TransformedTargetRegressor2(regressor=None, transformer="permute")
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        tt.fit(X, y)
        self.assertIn("TransformedTargetRegressor2", str(tt))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4,))
        sc = tt.score(X, y)
        self.assertLesser(sc, 1.0)

    def test_target_classifier(self):
        tt = TransformedTargetClassifier2(classifier=None, transformer="permute")
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.array([0, 0, 1, 1], dtype=int)
        tt.fit(X, y)
        self.assertIn("TransformedTargetClassifier2", str(tt))
        coef = tt.classifier_.coef_
        self.assertEqual(coef.shape, (1, 1))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4,))
        sc = tt.score(X, y)
        self.assertLesser(sc, 1.0)

    def test_target_classifier_proba(self):
        tt = TransformedTargetClassifier2(classifier=None, transformer="permute")
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
        tt = TransformedTargetClassifier2(classifier=None, transformer="permute")
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
        trans = FunctionReciprocalTransformer("log")
        tt = TransformedTargetRegressor2(regressor=None, transformer=trans)
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        tt.fit(X, y)
        self.assertIn("TransformedTargetRegressor2", str(tt))
        coef = tt.regressor_.coef_
        self.assertEqualArray(coef, numpy.array([2], dtype=float))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4,))
        sc = tt.score(X, y)
        self.assertLesser(sc, 1.0)

    def test_target_classifier_any(self):
        trans = FunctionReciprocalTransformer("log")
        tt = TransformedTargetClassifier2(classifier=None, transformer=trans)
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        tt.fit(X, y)
        self.assertIn("TransformedTargetClassifier2", str(tt))
        yp = tt.predict(X)
        self.assertEqual(yp.shape, (4,))

    def test_target_classifier_permute(self):
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.array([0, 0, 1, 1], dtype=int)

        log = LogisticRegression()
        log.fit(X, y)
        sc = log.score(X, y)

        tt = TransformedTargetClassifier2(classifier=None, transformer="permute")
        tt.fit(X, y)
        sc2 = tt.score(X, y)
        self.assertEqual(sc, sc2)

    @ignore_warnings(category=(ConvergenceWarning,))
    def test_target_classifier_permute_iris(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)

        log = LogisticRegression(n_jobs=1)
        log.fit(X_train, y_train)
        sc = log.score(X_test, y_test)
        r2 = r2_score(y_test, log.predict(X_test))

        for _ in range(10):
            TransformedTargetClassifier2(classifier=None, transformer="permute")
            tt = TransformedTargetClassifier2(
                classifier=LogisticRegression(n_jobs=1), transformer="permute"
            )
            try:
                tt.fit(X_train, y_train)
            except AttributeError as e:
                if compare_module_version(sklver, "0.24") < 0:
                    return
                raise e
            sc2 = tt.score(X_test, y_test)
            self.assertEqual(sc, sc2)
            r22 = r2_score(y_test, tt.predict(X_test))
            self.assertEqual(r2, r22)


if __name__ == "__main__":
    unittest.main()
