import unittest
import datetime
import warnings
import sys
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.timeseries.datasets import artificial_data
from mlinsights.timeseries.agg import aggregate_timeseries
from mlinsights.timeseries.plotting import plot_week_timeseries


class TestPlotTimeSeries(ExtTestCase):
    @unittest.skipIf(
        sys.platform == "win32" and __name__ != "__main__",
        reason="issue with matplotlib",
    )
    def test_plot_data(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            if "generated new fontManager" in str(e):
                warnings.warn(e)
                return
            raise e
        dt1 = datetime.datetime(2019, 8, 1)
        dt2 = datetime.datetime(2019, 8, 15)
        data = artificial_data(dt1, dt2, minutes=15)
        agg = aggregate_timeseries(data, per="week")
        ax = plot_week_timeseries(
            agg["weektime"],
            agg["y"],
            label="y",
            value2=agg["y"] / 2,
            label2="y/2",
            normalise=False,
        )
        self.assertNotEmpty(ax)
        if __name__ == "__main__":
            plt.show()
        plt.clf()


if __name__ == "__main__":
    unittest.main()
