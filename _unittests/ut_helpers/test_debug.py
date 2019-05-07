# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy.random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from pyquickhelper.pycode import ExtTestCase
from mlinsights.helpers.pipeline import alter_pipeline_for_debugging, enumerate_pipeline_models


class TestStr(ExtTestCase):

    def test_union_features(self):
        data = numpy.random.randn(4, 5)
        y = numpy.random.randn(4)
        model = Pipeline([('scaler1', StandardScaler()),
                          ('union', FeatureUnion([
                              ('scaler2', StandardScaler()),
                              ('scaler3', MinMaxScaler())])),
                          ('lr', LinearRegression())])

        model.fit(data, y)
        alter_pipeline_for_debugging(model)
        model.predict(data)
        for model_ in enumerate_pipeline_models(model):
            model = model_[1]
            if hasattr(model, '_debug'):
                text = str(model._debug)  # pylint: disable=W0212
                self.assertNotIn(" object at 0x", text)
                self.assertIn(") -> (", text)
            else:
                raise Exception("should not be the case")


if __name__ == "__main__":
    unittest.main()
