import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.timeseries import build_ts_X_y, ARTimeSeriesRegressor


class TestArtTimeSeries(ExtTestCase):
    def test_base_parameters_split0(self):
        X = None
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = ARTimeSeriesRegressor(past=2)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        self.assertEqualArray(y[0:-2], nx[:, 0])
        self.assertEqualArray(y[1:-1], nx[:, 1])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny)
        self.assertEqualArray(weights[1:-1], nw)

    def test_base_parameters_split0_all(self):
        X = None
        y = numpy.arange(5).astype(numpy.float64) * 100
        weights = numpy.arange(5).astype(numpy.float64) * 1000
        bs = ARTimeSeriesRegressor(past=2)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights, same_rows=True)
        self.assertEqualArray(y[0:-2], nx[2:, 0])
        self.assertEqualArray(y[1:-1], nx[2:, 1])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny[2:])
        self.assertEqualArray(weights, nw)

    def test_base_parameters_split0_1(self):
        X = None
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) + 1000
        bs = ARTimeSeriesRegressor(past=1)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        self.assertEqual(nx.shape, (4, 1))
        self.assertEqualArray(y[0:-1], nx[:, 0])
        self.assertEqualArray(y[1:].reshape((4, 1)), ny)
        self.assertEqualArray(weights[:-1], nw)

    def test_base_parameters_split1(self):
        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = ARTimeSeriesRegressor(past=2)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        self.assertEqualArray(X[1:-1], nx[:, :2])
        self.assertEqualArray(y[0:-2], nx[:, 2])
        self.assertEqualArray(y[1:-1], nx[:, 3])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny)
        self.assertEqualArray(weights[1:-1], nw)

    def test_base_parameters_split2(self):
        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = ARTimeSeriesRegressor(past=2, delay2=3)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        self.assertEqualArray(X[1:-2], nx[:, :2])
        self.assertEqualArray(y[0:-3], nx[:, 2])
        self.assertEqualArray(y[1:-2], nx[:, 3])
        self.assertEqualArray(numpy.array([[200, 300], [300, 400]]), ny)
        self.assertEqualArray(weights[1:-2], nw)

    def test_base_parameters_split_all_0(self):
        X = None
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = ARTimeSeriesRegressor(past=2, use_all_past=True)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        self.assertEqualArray(y[0:-2], nx[:, 0])
        self.assertEqualArray(y[1:-1], nx[:, 1])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny)
        self.assertEqualArray(weights[1:-1], nw)

    def test_base_parameters_split_all_0_same(self):
        X = None
        y = numpy.arange(5).astype(numpy.float64) * 100
        weights = numpy.arange(5).astype(numpy.float64) * 1000
        bs = ARTimeSeriesRegressor(past=2, use_all_past=True)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights, same_rows=True)
        self.assertEqualArray(y[0:-2], nx[2:, 0])
        self.assertEqualArray(y[1:-1], nx[1:-1, 1])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny[2:])
        self.assertEqualArray(weights, nw)

    def test_base_parameters_split_all_1(self):
        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = ARTimeSeriesRegressor(past=2, use_all_past=True)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
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
        bs = ARTimeSeriesRegressor(past=2, delay2=3, use_all_past=True)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        self.assertEqualArray(X[0:-3], nx[:, :2])
        self.assertEqualArray(X[1:-2], nx[:, 2:4])
        self.assertEqualArray(y[0:-3], nx[:, 4])
        self.assertEqualArray(y[1:-2], nx[:, 5])
        self.assertEqualArray(numpy.array([[200, 300], [300, 400]]), ny)
        self.assertEqualArray(weights[1:-2], nw)

    def test_fit_predict(self):
        X = None
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = ARTimeSeriesRegressor(past=2)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        self.assertEqualArray(y[0:-2], nx[:, 0])
        self.assertEqualArray(y[1:-1], nx[:, 1])
        self.assertEqualArray(y[2:].reshape((3, 1)), ny)
        self.assertEqualArray(weights[1:-1], nw)
        nx = nx.astype(numpy.float64)
        ny = ny.astype(numpy.float64)
        bs.fit(nx, ny)
        pred = bs.predict(nx, ny)
        self.assertEqual(pred.shape, ny.shape)


if __name__ == "__main__":
    unittest.main()
