"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.timeseries.base import BaseTimeSeries


class TestBaseTimeSeries(ExtTestCase):

    def test_base_parameters_split1(self):
        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2)
        nx, ny, nw = bs.build_X_y(X, y, weights)
        self.assertEqualArray(X[1:-1], nx[:, :2])
        self.assertEqualArray(y[0:-2], nx[:, 2])
        self.assertEqualArray(y[1:-1], nx[:, 3])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny)
        self.assertEqualArray(weights[1:-1], nw)

    def test_base_parameters_split2(self):
        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2, delay2=3)
        nx, ny, nw = bs.build_X_y(X, y, weights)
        self.assertEqualArray(X[1:-2], nx[:, :2])
        self.assertEqualArray(y[0:-3], nx[:, 2])
        self.assertEqualArray(y[1:-2], nx[:, 3])
        self.assertEqualArray(numpy.array([[200, 300], [300, 400]]), ny)
        self.assertEqualArray(weights[1:-2], nw)

    def test_base_parameters_split_all_1(self):
        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2, use_all_past=True)
        nx, ny, nw = bs.build_X_y(X, y, weights)
        self.assertEqualArray(X[0:-2], nx[:, :2])
        self.assertEqualArray(X[1:-1], nx[:, 2:4])
        self.assertEqualArray(y[0:-2], nx[:, 4])
        self.assertEqualArray(y[1:-1], nx[:, 5])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny)
        self.assertEqualArray(weights[1:-1], nw)

    def test_base_parameters_split_all_2(self):
        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2, delay2=3, use_all_past=True)
        nx, ny, nw = bs.build_X_y(X, y, weights)
        self.assertEqualArray(X[0:-3], nx[:, :2])
        self.assertEqualArray(X[1:-2], nx[:, 2:4])
        self.assertEqualArray(y[0:-3], nx[:, 4])
        self.assertEqualArray(y[1:-2], nx[:, 5])
        self.assertEqualArray(numpy.array([[200, 300], [300, 400]]), ny)
        self.assertEqualArray(weights[1:-2], nw)


if __name__ == "__main__":
    unittest.main()
