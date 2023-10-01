# -*- coding: utf-8 -*-
import unittest
import numpy
from scipy import sparse
from scipy.sparse import random as sparse_random
from sklearn.preprocessing import PolynomialFeatures
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel import ExtendedFeatures


class TestExtendedFeatures(ExtTestCase):
    def test_multiply(self):
        x1 = numpy.arange(9.0).reshape((3, 3))
        x2 = numpy.arange(3.0).reshape((3, 1))
        r = numpy.multiply(x1, x2)
        exp = numpy.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [12.0, 14.0, 16.0]])
        self.assertEqual(r, exp)

    def test_polynomial_features(self):
        X1 = numpy.arange(6)[:, numpy.newaxis].astype(numpy.float64)
        P1 = numpy.hstack([numpy.ones_like(X1), X1, X1**2, X1**3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2)).astype(numpy.float64)
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack(
            [
                x1**0 * x2**0,
                x1**1 * x2**0,
                x1**0 * x2**1,
                x1**2 * x2**0,
                x1**1 * x2**1,
                x1**0 * x2**2,
            ]
        )
        deg2 = 2

        for deg, X, P in [(deg1, X1, P1), (deg2, X2, P2)]:
            poly = PolynomialFeatures(deg, include_bias=True)
            P_test = poly.fit_transform(X)
            self.assertEqual(P_test, P)
            names = poly.get_feature_names_out()

            ext = ExtendedFeatures(poly_degree=deg)
            e_test = ext.fit_transform(X)
            e_names = ext.get_feature_names_out()
            self.assertEqual(len(names), len(e_names))
            self.assertEqual(list(names), list(e_names))

            self.assertEqual(P_test, P)
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_slow(self):
        X1 = numpy.arange(6)[:, numpy.newaxis].astype(numpy.float64)
        P1 = numpy.hstack([numpy.ones_like(X1), X1, X1**2, X1**3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2)).astype(numpy.float64)
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack(
            [
                x1**0 * x2**0,
                x1**1 * x2**0,
                x1**0 * x2**1,
                x1**2 * x2**0,
                x1**1 * x2**1,
                x1**0 * x2**2,
            ]
        )
        deg2 = 2

        for deg, X, P in [(deg1, X1, P1), (deg2, X2, P2)]:
            poly = PolynomialFeatures(deg, include_bias=True)
            P_test = poly.fit_transform(X)
            self.assertEqual(P_test, P)
            names = poly.get_feature_names_out()

            ext = ExtendedFeatures(kind="poly-slow", poly_degree=deg)
            e_test = ext.fit_transform(X)
            e_names = ext.get_feature_names_out()
            self.assertEqual(len(names), len(e_names))
            self.assertEqual(list(names), list(e_names))

            self.assertEqual(P_test, P)
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_nobias_ionly(self):
        X1 = numpy.arange(6)[:, numpy.newaxis].astype(numpy.float64)
        P1 = numpy.hstack([numpy.ones_like(X1), X1, X1**2, X1**3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2)).astype(numpy.float64)
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack(
            [
                x1**0 * x2**0,
                x1**1 * x2**0,
                x1**0 * x2**1,
                x1**2 * x2**0,
                x1**1 * x2**1,
                x1**0 * x2**2,
            ]
        )
        deg2 = 2

        for deg, X, P in [(deg1, X1, P1), (deg2, X2, P2)]:
            fc = [1] if deg == 3 else [1, 2, 4]
            poly = PolynomialFeatures(deg, include_bias=False, interaction_only=True)
            P_test = poly.fit_transform(X)

            names = poly.get_feature_names_out()
            self.assertEqual(P_test, P[:, fc])

            ext = ExtendedFeatures(
                poly_degree=deg, poly_include_bias=False, poly_interaction_only=True
            )
            e_test = ext.fit_transform(X)

            e_names = ext.get_feature_names_out()
            self.assertEqual(len(names), len(e_names))
            self.assertEqual(list(names), list(e_names))

            self.assertEqual(P_test, P[:, fc])
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_nobias_ionly_slow(self):
        X1 = numpy.arange(6)[:, numpy.newaxis].astype(numpy.float64)
        P1 = numpy.hstack([numpy.ones_like(X1), X1, X1**2, X1**3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2)).astype(numpy.float64)
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack(
            [
                x1**0 * x2**0,
                x1**1 * x2**0,
                x1**0 * x2**1,
                x1**2 * x2**0,
                x1**1 * x2**1,
                x1**0 * x2**2,
            ]
        )
        deg2 = 2

        for deg, X, P in [(deg1, X1, P1), (deg2, X2, P2)]:
            fc = [1] if deg == 3 else [1, 2, 4]
            poly = PolynomialFeatures(deg, include_bias=False, interaction_only=True)
            P_test = poly.fit_transform(X)

            names = poly.get_feature_names_out()
            self.assertEqual(P_test, P[:, fc])

            ext = ExtendedFeatures(
                kind="poly-slow",
                poly_degree=deg,
                poly_include_bias=False,
                poly_interaction_only=True,
            )
            e_test = ext.fit_transform(X)

            e_names = ext.get_feature_names_out()
            self.assertEqual(len(names), len(e_names))
            self.assertEqual(list(names), list(e_names))

            self.assertEqual(P_test, P[:, fc])
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_bias_ionly(self):
        X1 = numpy.arange(6)[:, numpy.newaxis].astype(numpy.float64)
        P1 = numpy.hstack([numpy.ones_like(X1), X1, X1**2, X1**3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2)).astype(numpy.float64)
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack(
            [
                x1**0 * x2**0,
                x1**1 * x2**0,
                x1**0 * x2**1,
                x1**2 * x2**0,
                x1**1 * x2**1,
                x1**0 * x2**2,
            ]
        )
        deg2 = 2

        for deg, X, P in [(deg1, X1, P1), (deg2, X2, P2)]:
            fc = [0, 1] if deg == 3 else [0, 1, 2, 4]
            poly = PolynomialFeatures(deg, include_bias=True, interaction_only=True)
            P_test = poly.fit_transform(X)

            names = poly.get_feature_names_out()
            self.assertEqual(P_test, P[:, fc])

            ext = ExtendedFeatures(
                poly_degree=deg, poly_include_bias=True, poly_interaction_only=True
            )
            e_test = ext.fit_transform(X)

            e_names = ext.get_feature_names_out()
            self.assertEqual(len(names), len(e_names))
            self.assertEqual(list(names), list(e_names))

            self.assertEqual(P_test, P[:, fc])
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_bias_ionly_slow(self):
        X1 = numpy.arange(6)[:, numpy.newaxis].astype(numpy.float64)
        P1 = numpy.hstack([numpy.ones_like(X1), X1, X1**2, X1**3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2)).astype(numpy.float64)
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack(
            [
                x1**0 * x2**0,
                x1**1 * x2**0,
                x1**0 * x2**1,
                x1**2 * x2**0,
                x1**1 * x2**1,
                x1**0 * x2**2,
            ]
        )
        deg2 = 2

        for deg, X, P in [(deg1, X1, P1), (deg2, X2, P2)]:
            fc = [0, 1] if deg == 3 else [0, 1, 2, 4]
            poly = PolynomialFeatures(deg, include_bias=True, interaction_only=True)
            P_test = poly.fit_transform(X)

            names = poly.get_feature_names_out()
            self.assertEqual(P_test, P[:, fc])

            ext = ExtendedFeatures(
                kind="poly-slow",
                poly_degree=deg,
                poly_include_bias=True,
                poly_interaction_only=True,
            )
            e_test = ext.fit_transform(X)

            e_names = ext.get_feature_names_out()
            self.assertEqual(len(names), len(e_names))
            self.assertEqual(list(names), list(e_names))

            self.assertEqual(P_test, P[:, fc])
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_nobias(self):
        X1 = numpy.arange(6)[:, numpy.newaxis].astype(numpy.float64)
        P1 = numpy.hstack([numpy.ones_like(X1), X1, X1**2, X1**3])
        deg1 = 3

        X2 = numpy.arange(6).reshape((3, 2)).astype(numpy.float64)
        x1 = X2[:, :1]
        x2 = X2[:, 1:]
        P2 = numpy.hstack(
            [
                x1**0 * x2**0,
                x1**1 * x2**0,
                x1**0 * x2**1,
                x1**2 * x2**0,
                x1**1 * x2**1,
                x1**0 * x2**2,
            ]
        )
        deg2 = 2

        for deg, X, P in [(deg1, X1, P1), (deg2, X2, P2)]:
            poly = PolynomialFeatures(deg, include_bias=False)
            P_test = poly.fit_transform(X)
            self.assertEqual(P_test, P[:, 1:])
            names = poly.get_feature_names_out()

            ext = ExtendedFeatures(poly_degree=deg, poly_include_bias=False)
            e_test = ext.fit_transform(X)
            self.assertEqual(P_test, P[:, 1:])
            e_names = ext.get_feature_names_out()

            self.assertEqual(len(names), len(e_names))
            self.assertEqual(list(names), list(e_names))
            self.assertEqual(P_test.shape, e_test.shape)
            self.assertEqual(P_test, e_test)

    def test_polynomial_features_bigger(self):
        X = numpy.arange(30).reshape((5, 6)).astype(numpy.float64)
        for deg in (1, 2, 3, 4):
            poly = PolynomialFeatures(deg, include_bias=True)
            X_sk = poly.fit_transform(X)
            names_sk = poly.get_feature_names_out()

            ext = ExtendedFeatures(poly_degree=deg)
            X_ext = ext.fit_transform(X)

            inames = ["x%d" % i for i in range(0, X.shape[1])]
            names_ext = ext.get_feature_names_out(inames)

            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(list(names_sk), list(names_ext))

            names_ext = ext.get_feature_names_out()
            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(list(names_sk), list(names_ext))

            self.assertEqual(X_sk.shape, X_ext.shape)
            self.assertEqual(X_sk, X_ext)

    def test_polynomial_features_bigger_ionly(self):
        X = numpy.arange(30).reshape((5, 6)).astype(numpy.float64)
        for deg in (1, 2, 3, 4, 5):
            poly = PolynomialFeatures(deg, include_bias=True, interaction_only=True)
            X_sk = poly.fit_transform(X)
            names_sk = poly.get_feature_names_out()

            ext = ExtendedFeatures(
                poly_degree=deg, poly_include_bias=True, poly_interaction_only=True
            )
            X_ext = ext.fit_transform(X)

            inames = ["x%d" % i for i in range(0, X.shape[1])]
            names_ext = ext.get_feature_names_out(inames)

            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(list(names_sk), list(names_ext))

            names_ext = ext.get_feature_names_out()
            self.assertEqual(len(names_sk), len(names_ext))
            self.assertEqual(list(names_sk), list(names_ext))

            self.assertEqual(X_sk.shape, X_ext.shape)
            self.assertEqual(X_sk, X_ext)

    @unittest.skip(reason="sparse not implemented for polynomial features")
    def test_polynomial_features_sparse(self):
        dtype = numpy.float64
        rng = numpy.random.RandomState(0)
        X = rng.randint(0, 2, (100, 2)).astype(numpy.float64)
        X_sparse = sparse.csr_matrix(X)

        est = PolynomialFeatures(2)
        Xt_sparse = est.fit_transform(X_sparse.astype(dtype))
        Xt_dense = est.fit_transform(X.astype(dtype))

        self.assertIsInstance(Xt_sparse, (sparse.csc_matrix, sparse.csr_matrix))
        self.assertEqual(Xt_sparse.dtype, Xt_dense.dtype)
        self.assertEqual(Xt_sparse.A, Xt_dense)

        est = ExtendedFeatures(poly_degree=2)
        Xt_sparse = est.fit_transform(X_sparse.astype(dtype))
        Xt_dense = est.fit_transform(X.astype(dtype))

        self.assertIsInstance(Xt_sparse, sparse.csc_matrix)
        self.assertEqual(Xt_sparse.dtype, Xt_dense.dtype)
        self.assertEqual(Xt_sparse.A, Xt_dense)

    def polynomial_features_csr_X_zero_row(self, zero_row_index, deg, interaction_only):
        X_csr = sparse_random(3, 10, 1.0, random_state=0).tocsr()
        X_csr[zero_row_index, :] = 0.0
        X = X_csr.toarray()
        est = ExtendedFeatures(
            poly_degree=deg,
            poly_include_bias=False,
            poly_interaction_only=interaction_only,
        )
        est.fit(X)
        poly = PolynomialFeatures(
            degree=deg, include_bias=False, interaction_only=interaction_only
        )
        poly.fit(X)
        self.assertEqual(
            list(poly.get_feature_names_out()), list(est.get_feature_names_out())
        )
        Xt_dense1 = est.fit_transform(X)
        Xt_dense2 = poly.fit_transform(X)
        self.assertEqual(Xt_dense1, Xt_dense2)

    def test_polynomial_features_bug(self):
        for p in [
            (0, 3, True),
            (0, 2, True),
            (1, 2, True),
            (2, 2, True),
            (1, 3, True),
            (2, 3, True),
            (0, 2, False),
            (1, 2, False),
            (2, 2, False),
            (0, 3, False),
            (1, 3, False),
            (2, 3, False),
        ]:
            self.polynomial_features_csr_X_zero_row(*list(p))


if __name__ == "__main__":
    unittest.main()
