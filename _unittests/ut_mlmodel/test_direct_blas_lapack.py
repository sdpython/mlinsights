# -*- coding: utf-8 -*-
import unittest
import numpy
from scipy.linalg.lapack import dgelss as scipy_dgelss
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel.direct_blas_lapack import dgelss


class TestDirectBlasLapack(ExtTestCase):
    def test_dgels0(self):
        A = numpy.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        C = numpy.array([[-1.0, 2.0]])
        B = numpy.matmul(A, C.T)

        ____, x, ___, __, _, info = scipy_dgelss(A, B)
        self.assertEqualArray(x.ravel()[:2], C.ravel(), atol=1e-8)
        A = A.T.copy()
        info = dgelss(A, B)
        self.assertEqual(info, 0)
        self.assertEqual(B.ravel()[:2], x.ravel()[:2])

    def test_dgels01(self):
        A = numpy.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        C = numpy.array([[-1.0, 2.0]])
        B = numpy.matmul(A, C.T)
        C[0, 0] = -0.9

        ____, x, ___, __, _, info = scipy_dgelss(A, B)
        A = A.T.copy()
        info = dgelss(A, B)
        self.assertEqual(info, 0)
        self.assertEqual(B.ravel()[:2], x.ravel()[:2])

    def test_dgels1(self):
        A = numpy.array([[10.0, 1.0], [12.0, 1.0], [13.0, 1]])
        B = numpy.array([[20.0, 22.0, 23.0]]).T
        ____, x, ___, __, _, info = scipy_dgelss(A, B)
        A = A.T.copy()
        info = dgelss(A, B)
        self.assertEqual(info, 0)
        self.assertEqual(B.ravel()[:2], x.ravel()[:2])


if __name__ == "__main__":
    unittest.main()
