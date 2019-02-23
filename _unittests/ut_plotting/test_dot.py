# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import pandas
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pyquickhelper.pycode import ExtTestCase


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from src.mlinsights.plotting import pipeline2dot


class TestDot(ExtTestCase):

    def test_dot_df(self):
        iris = datasets.load_iris()
        X = iris.data[:, :4]
        df = pandas.DataFrame(X)
        df.columns = ["X1", "X2", "X3", "X4"]
        clf = LogisticRegression()
        dot = pipeline2dot(clf, df)
        self.assertIn("digraph{", dot)
        self.assertIn("PredictedLabel|", dot)

    def test_dot_array(self):
        iris = datasets.load_iris()
        X = iris.data[:, :4]
        clf = LogisticRegression()
        dot = pipeline2dot(clf, X)
        self.assertIn("digraph{", dot)
        self.assertIn("PredictedLabel|", dot)

    def test_dot_list(self):
        clf = LogisticRegression()
        dot = pipeline2dot(clf, ['X1', 'X2'])
        self.assertIn("digraph{", dot)
        self.assertIn("PredictedLabel|", dot)

    def test_dot_list_reg(self):
        clf = LinearRegression()
        dot = pipeline2dot(clf, ['X1', 'X2'])
        self.assertIn("digraph{", dot)
        self.assertIn("Prediction", dot)
        self.assertIn("LinearRegression", dot)

    def test_dot_list_tr(self):
        clf = StandardScaler()
        dot = pipeline2dot(clf, ['X1', 'X2'])
        self.assertIn("digraph{", dot)
        self.assertIn("StandardScaler", dot)

    def test_pipeline(self):
        columns = ['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare',
                   'cabin', 'embarked', 'boat', 'body', 'home.dest']

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
        dot = pipeline2dot(clf, columns)
        self.assertIn("digraph{", dot)
        self.assertIn("StandardScaler", dot)

    def test_union_features(self):
        columns = ['X', 'Y']
        model = Pipeline([('scaler1', StandardScaler()),
                          ('union', FeatureUnion([
                              ('scaler2', StandardScaler()),
                              ('scaler3', MinMaxScaler())]))])
        dot = pipeline2dot(model, columns)
        self.assertIn("digraph{", dot)
        self.assertIn("StandardScaler", dot)
        self.assertIn("MinMaxScaler", dot)


if __name__ == "__main__":
    unittest.main()
