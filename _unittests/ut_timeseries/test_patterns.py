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
        names[:] = "A"
        for i in range(1, 20):
            names[i::20] = "BCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        self.assertRaise(
            lambda: find_ts_group_pattern(data["time"], data["y"], names), TypeError
        )
        clusters, dists = find_ts_group_pattern(
            data["time"].values, data["y"].values, names
        )
        self.assertEqual(clusters.shape[0], dists.shape[0])
        self.assertEqual(8, dists.shape[1])

    def test_clusters_norm(self):
        dt1 = datetime.datetime(2018, 8, 1)
        dt2 = datetime.datetime(2019, 8, 15)
        data = artificial_data(dt1, dt2, minutes=15)
        names = numpy.empty(data.shape[0], dtype=str)
        names[:] = "A"
        for i in range(1, 20):
            names[i::20] = "BCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        clusters, dists = find_ts_group_pattern(
            data["time"].values, data["y"].values, names, agg="norm"
        )
        self.assertEqual(clusters.shape[0], dists.shape[0])
        self.assertEqual(8, dists.shape[1])

    def test_clusters_subset(self):
        dt1 = datetime.datetime(2018, 8, 1)
        dt2 = datetime.datetime(2019, 8, 15)
        data = artificial_data(dt1, dt2, minutes=15)
        names = numpy.empty(data.shape[0], dtype=str)
        names[:] = "A"
        for i in range(1, 20):
            names[i::20] = "BCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        clusters, dists = find_ts_group_pattern(
            data["time"].values,
            data["y"].values,
            names,
            agg="norm",
            name_subset=list("BCDEFGHIJKL"),
        )
        self.assertEqual(clusters.shape[0], dists.shape[0])
        self.assertEqual(8, dists.shape[1])

    def test_clusters2(self):
        dt1 = datetime.datetime(2018, 8, 1)
        dt2 = datetime.datetime(2019, 8, 15)
        data = artificial_data(dt1, dt2, minutes=15)
        data["y2"] = data["y"] + 1.0
        names = numpy.empty(data.shape[0], dtype=str)
        names[:] = "A"
        for i in range(1, 20):
            names[i::20] = "BCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        clusters, dists = find_ts_group_pattern(
            data["time"].values, data[["y", "y2"]].values, names
        )
        self.assertEqual(clusters.shape[0], dists.shape[0])
        self.assertEqual(8, dists.shape[1])


if __name__ == "__main__":
    unittest.main()
