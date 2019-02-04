# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

from src.mlinsights.mlmodel import TransferTransformer
from src.mlinsights.mlmodel import test_sklearn_pickle, test_sklearn_clone


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

    def test_transfer_transformer_cloned0(self):
        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        norm = StandardScaler()
        norm.fit(X)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=False)
        tr1.fit(X)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=False)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        tr1.fit(X)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)

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
        test_sklearn_pickle(lambda: pipe, X, Y)

    def test_transfer_transformer_clone(self):

        X = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        Y = numpy.array([1., 1.1, 1.2, 10, 1.4])
        norm = StandardScaler()
        norm.fit(X)
        X2 = norm.transform(X)

        clr = LinearRegression()
        clr.fit(X2, Y)

        tr1 = TransferTransformer(norm, copy_estimator=False)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        tr1.fit(X)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        tr1.fit(X)
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)

        tr1 = TransferTransformer(norm, copy_estimator=True)
        tr2 = TransferTransformer(clr, copy_estimator=True)
        pipe = make_pipeline(tr1, tr2)
        pipe.fit(X)

        self.maxDiff = None
        test_sklearn_clone(lambda: tr1, ext=self, copy_fitted=True)
        test_sklearn_clone(lambda: tr2, ext=self, copy_fitted=True)
        test_sklearn_clone(lambda: pipe, ext=self, copy_fitted=True)


if __name__ == "__main__":
    unittest.main()
