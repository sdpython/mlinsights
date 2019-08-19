"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.timeseries import build_ts_X_y
from mlinsights.timeseries.base import BaseTimeSeries
from mlinsights.timeseries.preprocessing import TimeSeriesDifference


class TestPreprocessingTimeSeries(ExtTestCase):

    def test_base_parameters_split0(self):
        X = numpy.arange(20).reshape((10, 2))
        y = numpy.arange(10) * 100
        bs = BaseTimeSeries(past=2)
        nx, ny, _ = build_ts_X_y(bs, X, y)
        for d in range(0, 5):
            proc = TimeSeriesDifference(d)
            proc.fit(nx ,ny)
            px, py = proc.transform(nx, ny)
            self.assertEqualArray(px[-1, :], nx[-1, :])
            rev = proc.get_fct_inv()
            ppx, ppy = rev.transform(px, py)
            self.assertEqualArray(nx, ppx)
            self.assertEqualArray(ny, ppy)
            


if __name__ == "__main__":
    unittest.main()
