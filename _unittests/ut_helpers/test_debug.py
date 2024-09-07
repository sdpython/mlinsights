import unittest
import numpy.random
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.helpers.pipeline import (
    alter_pipeline_for_debugging,
    enumerate_pipeline_models,
)


class TestDebug(ExtTestCase):
    def test_union_features_reg(self):
        data = numpy.random.randn(4, 5)
        y = numpy.random.randn(4)
        model = Pipeline(
            [
                ("scaler1", StandardScaler()),
                (
                    "union",
                    FeatureUnion(
                        [("scaler2", StandardScaler()), ("scaler3", MinMaxScaler())]
                    ),
                ),
                ("lr", LinearRegression()),
            ]
        )

        model.fit(data, y)
        alter_pipeline_for_debugging(model)
        model.predict(data)
        for model_ in enumerate_pipeline_models(model):
            model = model_[1]
            if hasattr(model, "_debug"):
                text = str(model._debug)
                self.assertNotIn(" object at 0x", text)
                self.assertIn(") -> (", text)
            else:
                raise AssertionError("should not be the case")

    def test_union_features_cl(self):
        data = numpy.random.randn(4, 5)
        y = numpy.array([1, 1, 0, 0], dtype=numpy.int64)
        model = Pipeline(
            [
                ("scaler1", StandardScaler()),
                (
                    "union",
                    FeatureUnion(
                        [("scaler2", StandardScaler()), ("scaler3", MinMaxScaler())]
                    ),
                ),
                ("lr", LogisticRegression()),
            ]
        )

        model.fit(data, y)
        alter_pipeline_for_debugging(model)
        model.predict_proba(data)
        model.predict(data)
        for model_ in enumerate_pipeline_models(model):
            model = model_[1]
            if hasattr(model, "_debug"):
                text = str(model._debug)
                self.assertNotIn(" object at 0x", text)
                self.assertIn(") -> (", text)
            else:
                raise AssertionError("should not be the case")


if __name__ == "__main__":
    unittest.main()
