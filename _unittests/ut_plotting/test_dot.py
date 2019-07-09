# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from io import StringIO
from textwrap import dedent
import pandas
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pyquickhelper.pycode import ExtTestCase
from mlinsights.plotting import pipeline2dot


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

    def test_onehotencoder_dot(self):
        data = dedent("""
            date,value,notrend,trend,weekday,lag1,lag2,lag3,lag4,lag5,lag6,lag7,lag8
            2017-07-10 13:27:04.669830,0.003463591425601385,0.0004596547917981044,0.0030039366338032807,
            ###0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            2017-07-11 13:27:04.669830,0.004411953385609647,0.001342107238927262,0.003069846146682385,1,
            ###0.003463591425601385,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            2017-07-12 13:27:04.669830,0.004277700876279705,0.0011426168863444912,0.0031350839899352135,2,
            ###0.004411953385609647,0.003463591425601385,0.0,0.0,0.0,0.0,0.0,0.0
            2017-07-13 13:27:04.669830,0.006078151848127084,0.0028784976490072987,0.003199654199119785,3,
            ###0.004277700876279705,0.004411953385609647,0.003463591425601385,0.0,0.0,0.0,0.0,0.0
            2017-07-14 13:27:04.669830,0.006336617719481035,0.003073056920386795,0.0032635607990942395,
            ###4,0.006078151848127084,0.004277700876279705,0.004411953385609647,0.003463591425601385,0.0,0.0,0.0,0.0
            2017-07-15 13:27:04.669830,0.008716378909294038,0.0053895711052771985,0.0033268078040168394,5,
            ###0.006336617719481035,0.006078151848127084,0.004277700876279705,0.004411953385609647,0.003463591425601385,0.0,0.0,0.0
            2017-07-17 13:27:04.669830,0.0035533180858140765,0.00010197905397394454,0.003451339031840132,0,
            ###0.008716378909294038,0.006336617719481035,0.006078151848127084,0.004277700876279705,0.004411953385609647,0.003463591425601385,0.0,0.0
            2017-07-18 13:27:04.669830,0.0038464710972236286,0.0003338398676656705,0.003512631229557958,
            ###1,0.0035533180858140765,0.008716378909294038,0.006336617719481035,0.006078151848127084,0.004277700876279705,
            ###0.004411953385609647,0.003463591425601385,0.0
            2017-07-19 13:27:04.669830,0.004200435956007872,0.0006271561741496745,0.003573279781858197,2,0.0038464710972236286,
            ###0.0035533180858140765,0.008716378909294038,0.006336617719481035,0.006078151848127084,0.004277700876279705,
            ###0.004411953385609647,0.003463591425601385
            2017-07-20 13:27:04.669830,0.004773874566436903,0.0011405859170371827,0.00363328864939972,3,0.004200435956007872,
            ###0.0038464710972236286,0.0035533180858140765,0.008716378909294038,0.006336617719481035,0.006078151848127084,
            ###0.004277700876279705,0.004411953385609647
            2017-07-21 13:27:04.669830,0.005866058541412791,0.00217339675927127,0.0036926617821415207,4,0.004773874566436903,
            ###0.004200435956007872,0.0038464710972236286,0.0035533180858140765,0.008716378909294038,0.006336617719481035,
            ###0.006078151848127084,0.004277700876279705
            """).replace("\n###", "")
        df = pandas.read_csv(StringIO(data))
        cols = ['lag1', 'lag2', 'lag3',
                'lag4', 'lag5', 'lag6', 'lag7', 'lag8']
        model = make_pipeline(
            make_pipeline(
                ColumnTransformer(
                    [('pass', "passthrough", cols),
                     ("dummies", OneHotEncoder(), ["weekday"])]),
                PCA(n_components=2)),
            LinearRegression())
        train_cols = cols + ['weekday']
        model.fit(df, df[train_cols])
        dot = pipeline2dot(model, df)
        self.assertIn('label="Identity"', dot)


if __name__ == "__main__":
    unittest.main()
