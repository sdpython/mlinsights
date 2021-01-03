"""
@brief      test log(time=2s)
"""
import unittest
import pickle
from io import BytesIO
import pandas
from sklearn import __version__ as sklver
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper import compare_module_version
from mlinsights.sklapi import SkBaseTransformLearner


class TestSklearnConvert(ExtTestCase):

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_with_two_classifiers(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        conv = SkBaseTransformLearner(LogisticRegression(n_jobs=1))
        pipe = make_pipeline(conv, DecisionTreeClassifier())
        try:
            pipe.fit(X_train, y_train)
        except AttributeError as e:
            if compare_module_version(sklver, "0.24") < 0:
                return
            raise e
        pred = pipe.predict(X_test)
        score = accuracy_score(y_test, pred)
        self.assertGreater(score, 0.8)
        score2 = pipe.score(X_test, y_test)
        self.assertEqual(score, score2)
        rp = repr(conv)
        self.assertStartsWith(
            'SkBaseTransformLearner(model=LogisticRegression(', rp)

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_with_callable(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        tmod = LogisticRegression(n_jobs=1)
        conv = SkBaseTransformLearner(tmod, method=tmod.decision_function)
        pipe = make_pipeline(conv, DecisionTreeClassifier())
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = accuracy_score(y_test, pred)
        self.assertGreater(score, 0.8)
        score2 = pipe.score(X_test, y_test)
        self.assertEqualFloat(score, score2, precision=1e-5)
        rp = repr(conv)
        self.assertStartsWith(
            'SkBaseTransformLearner(model=LogisticRegression(', rp)

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_with_two_regressors(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        conv = SkBaseTransformLearner(LinearRegression())
        pipe = make_pipeline(conv, DecisionTreeRegressor())
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = r2_score(y_test, pred)
        self.assertLesser(score, 1.)
        score2 = pipe.score(X_test, y_test)
        self.assertEqualFloat(score, score2, precision=1e-5)
        rp = repr(conv)
        self.assertStartsWith(
            'SkBaseTransformLearner(model=LinearRegression(', rp)

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_with_params(self):
        conv = SkBaseTransformLearner(LinearRegression(normalize=True))
        pipe = make_pipeline(conv, DecisionTreeRegressor())
        pars = pipe.get_params()
        self.assertIn('skbasetransformlearner__model__fit_intercept', pars)
        self.assertEqual(
            pars['skbasetransformlearner__model__normalize'], True)
        conv = SkBaseTransformLearner(LinearRegression(normalize=True))
        pipe = make_pipeline(conv, DecisionTreeRegressor())
        pipe.set_params(**pars)
        pars = pipe.get_params()
        self.assertIn('skbasetransformlearner__model__fit_intercept', pars)
        self.assertEqual(
            pars['skbasetransformlearner__model__normalize'], True)

    @ignore_warnings(ConvergenceWarning)
    def test_pickle(self):
        df = pandas.DataFrame(dict(y=[0, 1, 0, 1, 0, 1, 0, 1],
                                   X1=[0.5, 0.6, 0.52, 0.62,
                                       0.5, 0.6, 0.51, 0.61],
                                   X2=[0.5, 0.6, 0.7, 0.5, 1.5, 1.6, 1.7, 1.8]))
        X = df.drop('y', axis=1)
        y = df['y']
        model = SkBaseTransformLearner(LinearRegression(normalize=True))
        model.fit(X, y)

        pred = model.transform(X)

        st = BytesIO()
        pickle.dump(model, st)
        st = BytesIO(st.getvalue())
        rec = pickle.load(st)
        pred2 = rec.transform(X)
        self.assertEqualArray(pred, pred2)

    @ignore_warnings(ConvergenceWarning)
    def test_grid(self):
        df = pandas.DataFrame(dict(y=[0, 1, 0, 1, 0, 1, 0, 1],
                                   X1=[0.5, 0.6, 0.52, 0.62,
                                       0.5, 0.6, 0.51, 0.61],
                                   X2=[0.5, 0.6, 0.7, 0.5, 1.5, 1.6, 1.7, 1.8]))
        X = df.drop('y', axis=1)
        y = df['y']
        model = make_pipeline(SkBaseTransformLearner(LinearRegression(normalize=True)),
                              LogisticRegression())
        res = model.get_params(True)
        self.assertGreater(len(res), 0)

        parameters = {
            'skbasetransformlearner__model__fit_intercept': [False, True]}
        clf = GridSearchCV(model, parameters, cv=3)
        clf.fit(X, y)

        pred = clf.predict(X)
        self.assertEqualArray(y, pred)


if __name__ == "__main__":
    unittest.main()
