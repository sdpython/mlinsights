"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.timeseries import build_ts_X_y, ARTimeSeriesRegressor


class TestDummyTimeSeries(ExtTestCase):

    def test_dummy_timesieres_regressor_2(self):
        X = None
        y = numpy.arange(10)
        bs = ARTimeSeriesRegressor(past=2)
        nx, ny, nw = build_ts_X_y(bs, X, y)
        self.assertEmpty(nw)
        bs.fit(nx, ny)
        np = bs.predict(nx)
        self.assertEqual(np.ravel(), numpy.arange(1, 9))

    def test_dummy_timesieres_regressor_1(self):
        X = None
        y = numpy.arange(10)
        bs = ARTimeSeriesRegressor(past=1)
        nx, ny, nw = build_ts_X_y(bs, X, y)
        self.assertEmpty(nw)
        bs.fit(nx, ny)
        np = bs.predict(nx)
        self.assertEqual(np.ravel(), numpy.arange(0, 9))

    def test_dummy_timesieres_regressor_score(self):
        X = None
        y = numpy.arange(10)
        bs = ARTimeSeriesRegressor(past=1)
        nx, ny, nw = build_ts_X_y(bs, X, y)
        self.assertEmpty(nw)
        bs.fit(nx, ny)
        np = bs.predict(nx)
        self.assertEqual(np.ravel(), numpy.arange(0, 9))
        sc = bs.score(nx, ny)
        self.assertEqual(sc, 1)
        sc = bs.score(nx, ny, numpy.ones((len(ny),), ) * 2)
        self.assertEqual(sc, 1)


if __name__ == "__main__":
    unittest.main()
