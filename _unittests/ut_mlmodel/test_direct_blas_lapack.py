# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
import numpy
from scipy.linalg.lapack import dgelss as scipy_dgelss  # pylint: disable=E0611
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel.direct_blas_lapack import dgelss  # pylint: disable=E0611, E0401


class TestDirectBlasLapack(ExtTestCase):

    def test_dgels0(self):
        A = numpy.array([[1., 1.], [2., 1.], [3., 1.]])
        C = numpy.array([[-1., 2.]])
        B = numpy.matmul(A, C.T)

        ____, x, ___, __, _, info = scipy_dgelss(A, B)
        self.assertEqual(x.ravel()[:2], C.ravel())
        A = A.T.copy()
        info = dgelss(A, B)
        self.assertEqual(info, 0)
        self.assertEqual(B.ravel()[:2], x.ravel()[:2])

    def test_dgels01(self):
        A = numpy.array([[1., 1.], [2., 1.], [3., 1.]])
        C = numpy.array([[-1., 2.]])
        B = numpy.matmul(A, C.T)
        C[0, 0] = -0.9

        ____, x, ___, __, _, info = scipy_dgelss(A, B)
        A = A.T.copy()
        info = dgelss(A, B)
        self.assertEqual(info, 0)
        self.assertEqual(B.ravel()[:2], x.ravel()[:2])

    def test_dgels1(self):
        A = numpy.array([[10., 1.], [12., 1.], [13., 1]])
        B = numpy.array([[20., 22., 23.]]).T
        ____, x, ___, __, _, info = scipy_dgelss(A, B)
        A = A.T.copy()
        info = dgelss(A, B)
        self.assertEqual(info, 0)
        self.assertEqual(B.ravel()[:2], x.ravel()[:2])


if __name__ == "__main__":
    unittest.main()
