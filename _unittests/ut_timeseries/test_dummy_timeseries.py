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


if __name__ == "__main__":
    unittest.main()
