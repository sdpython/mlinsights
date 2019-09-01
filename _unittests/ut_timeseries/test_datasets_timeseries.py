"""
@brief      test log(time=2s)
"""
import unittest
import datetime
from pyquickhelper.pycode import ExtTestCase
from mlinsights.timeseries.datasets import artificial_data


class TestDataSetsTimeSeries(ExtTestCase):

    def test_artificial_data(self):
        dt1 = datetime.datetime(2019, 8, 1)
        dt2 = datetime.datetime(2019, 9, 1)
        data = artificial_data(dt1, dt2, minutes=24 * 60)
        self.assertEqual(data.shape, (27, 2))
        data = artificial_data(dt1, dt2, minutes=60)
        self.assertEqual(data.shape, (297, 2))
        self.assertEqual(data.shape[0] * 1. / 27, data.shape[0] // 27)


if __name__ == "__main__":
    unittest.main()
