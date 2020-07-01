# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy.random
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from pyquickhelper.pycode import ExtTestCase
from mlinsights.helpers.parameters import format_value


class TestParameters(ExtTestCase):

    def test_format_value(self):
        self.assertEqual("3", format_value(3))
        self.assertEqual("'3'", format_value("3"))


if __name__ == "__main__":
    unittest.main()
