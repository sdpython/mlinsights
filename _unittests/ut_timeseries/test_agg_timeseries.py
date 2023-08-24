import unittest
import datetime
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.timeseries.datasets import artificial_data
from mlinsights.timeseries.agg import aggregate_timeseries


class TestAggTimeSeries(ExtTestCase):
    def test_agg_data(self):
        dt1 = datetime.datetime(2019, 8, 1)
        dt2 = datetime.datetime(2019, 8, 8)
        data = artificial_data(dt1, dt2, minutes=15)
        data["y"] = 1
        agg = aggregate_timeseries(data, per="week")
        self.assertEqual(agg.shape, (132, 2))
        self.assertEqual(agg["y"].min(), 2)
        self.assertEqual(agg["y"].max(), 2)

        dt1 = datetime.datetime(2019, 8, 1)
        dt2 = datetime.datetime(2019, 8, 15)
        data = artificial_data(dt1, dt2, minutes=15)
        data["y"] = 1
        agg = aggregate_timeseries(data, per="week")
        self.assertEqual(agg.shape, (132, 2))
        self.assertEqual(agg["y"].min(), 4)
        self.assertEqual(agg["y"].max(), 4)

        agg = aggregate_timeseries(data, per="month")
        self.assertEqual(agg.shape, (264, 2))
        self.assertEqual(agg["y"].min(), 2)
        self.assertEqual(agg["y"].max(), 2)


if __name__ == "__main__":
    unittest.main()
