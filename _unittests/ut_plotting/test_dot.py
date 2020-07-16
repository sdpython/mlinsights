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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pyquickhelper.pycode import ExtTestCase
from mlinsights.plotting import pipeline2dot, pipeline2str


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

    def test_pipeline_tr_small(self):

        buffer = """
            fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,color
            7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
            7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,red
            7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,red
            11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,white
            7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
            """.replace("            ", "")
        X_train = pandas.read_csv(StringIO(buffer)).drop("quality", axis=1)

        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("color", Pipeline([
                    ('one', "passthrough"),
                    ('select', ColumnTransformer(
                        [('sel1', 'passthrough', [0])]))
                ]), ['color']),
            ])),
        ])

        pipe.fit(X_train)
        dot = pipeline2dot(pipe, X_train)
        self.assertNotIn("i -> node2;", dot)

    def test_pipeline_tr(self):

        buffer = """
            fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,color
            7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
            7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,red
            7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,red
            11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,white
            7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
            """.replace("            ", "")
        X_train = pandas.read_csv(StringIO(buffer)).drop("quality", axis=1)

        numeric_features = [c for c in X_train if c != 'color']

        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("color", Pipeline([
                    ('one', OneHotEncoder()),
                    ('select', ColumnTransformer(
                        [('sel1', 'passthrough', [0])]))
                ]), ['color']),
                ("others", "passthrough", numeric_features)
            ])),
        ])

        pipe.fit(X_train)
        dot = pipeline2dot(pipe, X_train)
        self.assertIn("OneHotEncoder", dot)
        # self.assertIn("sch3:f10 -> node4;", dot)
        dots = pipeline2str(pipe)
        self.assertIn("OneHotEncoder", dots)
        self.assertIn('PassThrough(0)', dots)

    def test_pipeline_bug(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        pipe2 = Pipeline([
            ('multi', ColumnTransformer([
                ('c01', Normalizer(), [0, 1]),
                ('c23', MinMaxScaler(), [2, 3]),
            ])),
            ('pca', PCA()),
            ('lr', LogisticRegression())
        ])

        pipe2.fit(X, y)
        dot = pipeline2dot(pipe2, X)
        self.assertIn("MinMaxScaler", dot)
        self.assertNotIn("  0 -> node1;", dot)
        self.assertIn("  sch0:f0 -> node1;", dot)

    def test_pipeline_bug2(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        pipe2 = Pipeline([
            ('multi', ColumnTransformer([
                ('c01a', Normalizer(), [0, 1]),
                ('c23a', MinMaxScaler(), [2, 3]),
            ])),
            ('multi2', ColumnTransformer([
                ('c01b', Normalizer(), [0, 1]),
                ('c23b', MinMaxScaler(), [2, 3]),
            ])),
            ('pca', PCA()),
            ('lr', LogisticRegression())
        ])

        pipe2.fit(X, y)
        dot = pipeline2dot(pipe2, X)
        self.assertIn("MinMaxScaler", dot)
        self.assertNotIn("  0 -> node1;", dot)
        # self.assertNotIn("sch1:f0 -> node2;", dot)

    def test_pipeline_passthrough(self):

        data = pandas.DataFrame([
            dict(CAT1='a', CAT2='c', num1=0.5, num2=0.6, y=0),
            dict(CAT1='b', CAT2='d', num1=0.4, num2=0.8, y=1),
            dict(CAT1='a', CAT2='d', num1=0.5, num2=0.56, y=0),
            dict(CAT1='a', CAT2='d', num1=0.55, num2=0.56, y=1),
            dict(CAT1='a', CAT2='c', num1=0.35, num2=0.86, y=0),
            dict(CAT1='a', CAT2='c', num1=0.5, num2=0.68, y=1),
        ])

        cat_cols = ['CAT1', 'CAT2']
        train_data = data.drop('y', axis=1)

        # numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, cat_cols)],
            remainder='passthrough')
        pipe = Pipeline([('preprocess', preprocessor),
                         ('rf', RandomForestClassifier(n_estimators=2))])
        pipe.fit(train_data, data['y'])
        dot = pipeline2dot(pipe, train_data)
        self.assertIn("sch0:f2 ->", dot)
        self.assertNotIn("node3 -> sch3:f34;", dot)
        self.assertIn("node3 -> sch3:f0;", dot)
        self.assertIn("node3 -> sch3:f1;", dot)
        self.assertNotIn("node3 -> sch3:f2;", dot)


if __name__ == "__main__":
    # TestDot().test_pipeline_passthrough()
    unittest.main()
