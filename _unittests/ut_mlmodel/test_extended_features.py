# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import numpy
from scipy import sparse
from sklearn.preprocessing import PolynomialFeatures
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

from src.mlinsights.mlmodel import ExtendedFeatures


class TestExtendedFeatures(ExtTestCase):

    def test_multiply(self):
        x1 = numpy.arange(9.0).reshape((3, 3))
        x2 = numpy.arange(3.0).reshape((3, 1))
        r = numpy.multiply(x1, x2)
        exp = numpy.array([[0., 0., 0.], [3., 4., 5.], [12., 14., 16.]])
        self.assertEqual(r, exp)

    def test_polynomial_features(self):
        X1 = numpy.arange(6)[:, numpy.newaxis]
        P1 = numpy.hstack([numpy.ones_like(X1),
                           X1, X1 ** 2, X1 ** 3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2))
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack([x1 ** 0 * x2 ** 0,
                           x1 ** 1 * x2 ** 0,
                           x1 ** 0 * x2 ** 1,
                           x1 ** 2 * x2 ** 0,
                           x1 ** 1 * x2 ** 1,
                           x1 ** 0 * x2 ** 2])
        deg2 = 2

        for (deg, X, P) in [(deg1, X1, P1), (deg2, X2, P2)]:
            poly = PolynomialFeatures(deg, include_bias=True)
            P_test = poly.fit_transform(X)
            self.assertEqual(P_test, P)
            names = poly.get_feature_names()

            ext = ExtendedFeatures(poly_degree=deg)
            e_test = ext.fit_transform(X)
            self.assertEqual(P_test, P)
            e_names = ext.get_feature_names()

            self.assertEqual(len(names), len(e_names))
            self.assertEqual(names, e_names)
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_bigger(self):
        X = numpy.arange(15).reshape((5, 3))
        for deg in (1, 2, 3, 4):
            poly = PolynomialFeatures(deg, include_bias=True)
            X_sk = poly.fit_transform(X)
            names_sk = poly.get_feature_names()

            ext = ExtendedFeatures(poly_degree=deg)
            X_ext = ext.fit_transform(X)

            inames = ["x%d" % i for i in range(0, X.shape[1])]
            names_ext = ext.get_feature_names(inames)

            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(names_sk, names_ext)

            names_ext = ext.get_feature_names()
            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(names_sk, names_ext)

            self.assertEqual(X_sk.shape, X_ext.shape)
            self.assertEqual(X_sk, X_ext)

    def test_polynomial_features_sparse(self):
        dtype = numpy.float64
        rng = numpy.random.RandomState(0)  # pylint: disable=E1101
        X = rng.randint(0, 2, (100, 2))
        X_sparse = sparse.csr_matrix(X)

        est = PolynomialFeatures(2)
        Xt_sparse = est.fit_transform(X_sparse.astype(dtype))
        Xt_dense = est.fit_transform(X.astype(dtype))

        self.assertIsInstance(Xt_sparse, sparse.csc_matrix)
        self.assertEqual(Xt_sparse.dtype, Xt_dense.dtype)
        self.assertEqual(Xt_sparse.A, Xt_dense)

        est = ExtendedFeatures(poly_degree=2)
        Xt_sparse = est.fit_transform(X_sparse.astype(dtype))
        Xt_dense = est.fit_transform(X.astype(dtype))

        self.assertIsInstance(Xt_sparse, sparse.csc_matrix)
        self.assertEqual(Xt_sparse.dtype, Xt_dense.dtype)
        self.assertEqual(Xt_sparse.A, Xt_dense)

    def test_polynomial_features_bigger_transpose(self):
        X = numpy.arange(15).reshape((5, 3))
        for deg in (1, 2, 3, 4):
            poly = PolynomialFeatures(deg, include_bias=True)
            X_sk = poly.fit_transform(X)
            names_sk = poly.get_feature_names()

            ext = ExtendedFeatures(poly_degree=deg, poly_transpose=True)
            X_ext = ext.fit_transform(X)

            inames = ["x%d" % i for i in range(0, X.shape[1])]
            names_ext = ext.get_feature_names(inames)

            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(names_sk, names_ext)

            names_ext = ext.get_feature_names()
            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(names_sk, names_ext)

            self.assertEqual(X_sk.shape, X_ext.shape)
            self.assertEqual(X_sk, X_ext)


if __name__ == "__main__":
    unittest.main()
