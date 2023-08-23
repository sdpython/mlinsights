import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.timeseries.preprocessing import TimeSeriesDifference
from mlinsights.timeseries.dummies import DummyTimeSeriesRegressor


class TestDummyTimeSeries(ExtTestCase):
    def test_dummy_timesieres_regressor_2(self):
        X = None
        y = numpy.arange(10)
        bs = DummyTimeSeriesRegressor(past=2)
        self.assertRaise(lambda: bs.fit(X, y), TypeError)
        y = y.astype(numpy.float64)
        np = bs.predict(X, y)
        self.assertEqual(np.ravel()[2:], numpy.arange(1, 9))

    def test_dummy_timesieres_regressor_1(self):
        X = None
        y = numpy.arange(10)
        y = y.astype(numpy.float64)
        bs = DummyTimeSeriesRegressor(past=1)
        bs.fit(X, y)
        np = bs.predict(X, y)
        self.assertEqual(np.ravel()[1:], numpy.arange(0, 9))

    def test_dummy_timesieres_regressor_score(self):
        X = None
        y = numpy.arange(10)
        y = y.astype(numpy.float64)
        bs = DummyTimeSeriesRegressor(past=1)
        bs.fit(X, y)
        np = bs.predict(X, y)
        self.assertEqual(np.ravel()[1:], numpy.arange(0, 9))
        sc = bs.score(X, y)
        self.assertEqual(sc, 1)
        sc = bs.score(
            X,
            y,
            numpy.ones(
                (len(y),),
            )
            * 2,
        )
        self.assertEqual(sc, 1)

    def test_dummy_timeseries_regressor_1_diff(self):
        X = None
        y = numpy.arange(10).astype(numpy.float64)
        bs = DummyTimeSeriesRegressor(past=1, preprocessing=TimeSeriesDifference(1))
        bs.fit(X, y)
        self.assertRaise(
            lambda: bs.predict(X), (TypeError, RuntimeError)  # pylint: disable=E1120
        )
        for i in range(y.shape[0]):
            if i >= y.shape[0] - 2:
                self.assertRaise(lambda ii=i: bs.predict(None, y[ii:]), AssertionError)
            else:
                np = bs.predict(None, y[i:])
                self.assertEqual(np.shape[0] + 1, y[i:].shape[0])
        np = bs.predict(X, y).ravel()
        self.assertEqual(np[1:], numpy.arange(1, 9))
        self.assertTrue(numpy.isnan(np[0]))


if __name__ == "__main__":
    TestDummyTimeSeries().test_dummy_timesieres_regressor_score()
    unittest.main()
