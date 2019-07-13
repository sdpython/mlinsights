# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel.anmf_predictor import ApproximateNMFPredictor


class TestApproximateNMFPredictor(ExtTestCase):

    def test_anmf_predictor(self):
        mat = numpy.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                           [1, 0, 0, 0], [1, 0, 0, 0]], dtype=numpy.float64)
        mat[:mat.shape[1], :] += numpy.identity(mat.shape[1])

        mod = ApproximateNMFPredictor(n_components=2)
        mod.fit(mat)
        exp = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat))
        got = mod.predict(mat)
        sc1 = mean_squared_error(mat, exp)
        sc2 = mean_squared_error(mat, got)
        self.assertGreater(sc1, sc2)

        mat2 = numpy.array([[1, 1, 1, 1]], dtype=numpy.float64)
        exp2 = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat2))
        got2 = mod.predict(mat2)
        sc1 = mean_squared_error(mat2, exp2)
        sc2 = mean_squared_error(mat2, got2)
        self.assertGreater(sc1, sc2)

    def test_anmf_predictor_sparse(self):
        mat = numpy.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                           [1, 0, 0, 0], [1, 0, 0, 0]], dtype=numpy.float64)
        mat[:mat.shape[1], :] += numpy.identity(mat.shape[1])
        mat = csr_matrix(mat)

        mod = ApproximateNMFPredictor(n_components=2)
        mod.fit(mat)
        exp = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat))
        got = mod.predict(mat)
        sc1 = mean_squared_error(mat.todense(), exp)
        sc2 = mean_squared_error(mat.todense(), got)
        self.assertGreater(sc1, sc2)

        mat2 = numpy.array([[1, 1, 1, 1]], dtype=numpy.float64)
        exp2 = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat2))
        got2 = mod.predict(mat2)
        sc1 = mean_squared_error(mat2, exp2)
        sc2 = mean_squared_error(mat2, got2)
        self.assertGreater(sc1, sc2)

    def test_anmf_predictor_sparse_sparse(self):
        mat = numpy.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                           [1, 0, 0, 0], [1, 0, 0, 0]], dtype=numpy.float64)
        mat[:mat.shape[1], :] += numpy.identity(mat.shape[1])
        mat = csr_matrix(mat)

        mod = ApproximateNMFPredictor(n_components=2)
        mod.fit(mat)
        exp = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat))
        got = mod.predict(mat)
        sc1 = mean_squared_error(mat.todense(), exp)
        sc2 = mean_squared_error(mat.todense(), got)
        self.assertGreater(sc1, sc2)

        mat2 = numpy.array([[1, 1, 1, 1]], dtype=numpy.float64)
        mat2 = csr_matrix(mat2)
        exp2 = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat2))
        got2 = mod.predict(mat2)
        sc1 = mean_squared_error(mat2.todense(), exp2)
        sc2 = mean_squared_error(mat2.todense(), got2)
        self.assertGreater(sc1, sc2)

    def test_anmf_predictor_positive(self):
        mat = numpy.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                           [1, 0, 0, 0], [1, 0, 0, 0]], dtype=numpy.float64)
        mat[:mat.shape[1], :] += numpy.identity(mat.shape[1])

        mod = ApproximateNMFPredictor(n_components=2, force_positive=True)
        mod.fit(mat)
        exp = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat))
        got = mod.predict(mat)
        sc1 = mean_squared_error(mat, exp)
        sc2 = mean_squared_error(mat, got)
        self.assertGreater(sc1, sc2)
        mx = numpy.min(got)
        self.assertGreater(mx, 0)

        mat2 = numpy.array([[1, 1, 1, 1]], dtype=numpy.float64)
        exp2 = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat2))
        got2 = mod.predict(mat2)
        sc1 = mean_squared_error(mat2, exp2)
        sc2 = mean_squared_error(mat2, got2)
        self.assertGreater(sc1, sc2)
        mx = numpy.min(got2)
        self.assertGreater(mx, 0)

    def test_anmf_predictor_positive_sparse(self):
        mat = numpy.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                           [1, 0, 0, 0], [1, 0, 0, 0]], dtype=numpy.float64)
        mat[:mat.shape[1], :] += numpy.identity(mat.shape[1])
        mat = csr_matrix(mat)

        mod = ApproximateNMFPredictor(n_components=2, force_positive=True)
        mod.fit(mat)
        exp = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat))
        got = mod.predict(mat)
        sc1 = mean_squared_error(mat.todense(), exp)
        sc2 = mean_squared_error(mat.todense(), got)
        self.assertGreater(sc1, sc2)
        mx = numpy.min(got)
        self.assertGreater(mx, 0)

        mat2 = numpy.array([[1, 1, 1, 1]], dtype=numpy.float64)
        exp2 = mod.estimator_nmf_.inverse_transform(
            mod.estimator_nmf_.transform(mat2))
        got2 = mod.predict(mat2)
        sc1 = mean_squared_error(mat2, exp2)
        sc2 = mean_squared_error(mat2, got2)
        self.assertGreater(sc1, sc2)
        mx = numpy.min(got2)
        self.assertGreater(mx, 0)


if __name__ == "__main__":
    unittest.main()
