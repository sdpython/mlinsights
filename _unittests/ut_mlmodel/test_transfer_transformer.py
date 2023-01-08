# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel import TransferTransformer
from mlinsights.mlmodel import run_test_sklearn_pickle, run_test_sklearn_clone


class TestTransferTransformer(ExtTestCase):

    def test_transfer_transformer_diff(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LinearRegression()
        clr.fit(X2, Y)
        exp = clr.predict(X2)

        pipe = make_pipeline(TransferTransformer(norm),
                             TransferTransformer(clr))
        pipe.fit(X)
        got = pipe.transform(X)
        self.assertEqual(exp, got)

    def test_transfer_transformer_sample_weight(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        sw = numpy.array([1, 1, 1.5, 1.5, 1.])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LinearRegression()
        clr.fit(X2, Y)
        exp = clr.predict(X2)

        pipe = Pipeline(steps=[
            ('scaler', TransferTransformer(norm)),
            ('model', TransferTransformer(clr))])
        pipe.fit(X, model__sample_weight=sw)
        got = pipe.transform(X)
        self.assertEqual(exp, got)

    def test_transfer_transformer_logreg(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([0, 0, 1, 1, 1])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LogisticRegression()
        clr.fit(X2, Y)
        exp = clr.predict_proba(X2)

        pipe = make_pipeline(TransferTransformer(norm),
                             TransferTransformer(clr))
        pipe.fit(X)
        got = pipe.transform(X)
        self.assertEqual(exp, got)

    def test_transfer_transformer_decision_function(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([0, 0, 1, 1, 1])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LogisticRegression()
        clr.fit(X2, Y)
        exp = clr.decision_function(X2)

        pipe = make_pipeline(
            TransferTransformer(norm),
            TransferTransformer(clr, method="decision_function"))
        pipe.fit(X)
        got = pipe.transform(X)
        self.assertEqual(exp, got)

    def test_transfer_transformer_diff_trainable(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LinearRegression()
        clr.fit(X2, Y)
        exp = clr.predict(X2)

        pipe = make_pipeline(TransferTransformer(norm, trainable=True),
                             TransferTransformer(clr, trainable=True))
        pipe.fit(X, Y)
        got = pipe.transform(X)
        self.assertEqual(exp, got)

    def test_transfer_transformer_cloned0(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        norm = StandardScaler()
        norm.fit(X)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=False)
        tr1.fit(X)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=False)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        tr1.fit(X)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)

    def test_transfer_transformer_pickle(self):

        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LinearRegression()
        clr.fit(X2, Y)

        pipe = make_pipeline(TransferTransformer(norm),
                             TransferTransformer(clr))
        pipe.fit(X)
        run_test_sklearn_pickle(lambda: pipe, X, Y)

    def test_transfer_transformer_clone(self):

        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LinearRegression()
        clr.fit(X2, Y)

        tr1 = TransferTransformer(norm, copy_estimator=False)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        tr1.fit(X)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        tr1.fit(X)
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        tr2 = TransferTransformer(clr, copy_estimator=True)
        pipe = make_pipeline(tr1, tr2)
        pipe.fit(X)

        self.maxDiff = None
        run_test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        run_test_sklearn_clone(lambda: tr2, ext=self, copy_fitted=True)
        run_test_sklearn_clone(lambda: pipe, ext=self, copy_fitted=True)


if __name__ == "__main__":
    unittest.main()
