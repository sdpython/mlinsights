"""
@brief      test log(time=2s)
"""
import unittest
import datetime
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.timeseries.datasets import artificial_data
from mlinsights.timeseries.patterns import find_ts_group_pattern


class TestPatterns(ExtTestCase):

    def test_clusters(self):
        dt1 = datetime.datetime(2018, 8, 1)
        dt2 = datetime.datetime(2019, 8, 15)
        data = artificial_data(dt1, dt2, minutes=15)
        names = numpy.empty(data.shape[0], dtype=str)
        names[:] = 'A'
        for i in range(1, 20):
            names[i::20] = 'BCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
        clusters, dists = find_ts_group_pattern(data['time'], data['y'], names)
        self.assertEqual(clusters.shape[0], dists.shape[0])
        self.assertEqual(10, dists.shape[1])

    def test_clusters2(self):
        dt1 = datetime.datetime(2018, 8, 1)
        dt2 = datetime.datetime(2019, 8, 15)
        data = artificial_data(dt1, dt2, minutes=15)
        data['y2'] = data['y'] + 1.
        names = numpy.empty(data.shape[0], dtype=str)
        names[:] = 'A'
        for i in range(1, 20):
            names[i::20] = 'BCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
        clusters, dists = find_ts_group_pattern(
            data['time'], data[['y', 'y2']].values, names)
        self.assertEqual(clusters.shape[0], dists.shape[0])
        self.assertEqual(10, dists.shape[1])


if __name__ == "__main__":
    TestPatterns().test_clusters2()
    unittest.main()
