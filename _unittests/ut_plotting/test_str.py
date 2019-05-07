# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pyquickhelper.pycode import ExtTestCase
from mlinsights.plotting import pipeline2str


class TestStr(ExtTestCase):

    def test_pipeline(self):
        numeric_features = ['age', 'fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['embarked', 'sex', 'pclass']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ])

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(solver='lbfgs'))])
        text = pipeline2str(clf)
        self.assertIn("StandardScaler", text)
        self.assertIn("Pipeline(embarked,sex,pclass)", text)

    def test_union_features(self):
        model = Pipeline([('scaler1', StandardScaler()),
                          ('union', FeatureUnion([
                              ('scaler2', StandardScaler()),
                              ('scaler3', MinMaxScaler())]))])
        text = pipeline2str(model)
        self.assertIn("StandardScaler", text)
        self.assertIn("MinMaxScaler", text)


if __name__ == "__main__":
    unittest.main()
