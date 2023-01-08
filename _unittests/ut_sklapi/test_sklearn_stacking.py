"""
@brief      test log(time=5s)
"""
import os
import unittest
from io import BytesIO
import pickle
import warnings
import pandas
from numpy.random import permutation
from sklearn import __version__ as sklver
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Normalizer, MinMaxScaler
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper import compare_module_version
from mlinsights.sklapi import SkBaseTransformStacking

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from sklearn.ensemble import RandomForestClassifier  # pylint: disable=C0412


def load_wines_dataset(shuffle=False):
    this = os.path.split(os.path.abspath(__file__))[0]
    data = os.path.join(this, "data")
    name = os.path.join(data, "wines-quality.csv")
    df = pandas.read_csv(name)
    if shuffle:
        df = df.reset_index(drop=True)
        ind = permutation(df.index)
        df = df.iloc[ind, :].reset_index(drop=True)
    return df


class TestSklearnStacking(ExtTestCase):

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_with_two_classifiers(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        conv = SkBaseTransformStacking(
            [LogisticRegression(n_jobs=1), DecisionTreeClassifier()])
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
            'SkBaseTransformStacking([LogisticRegression(', rp)

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_with_two_transforms(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        conv = SkBaseTransformStacking(
            [Normalizer(), MinMaxScaler()])
        pipe = make_pipeline(conv, DecisionTreeClassifier())
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = accuracy_score(y_test, pred)
        self.assertGreater(score, 0.8)
        score2 = pipe.score(X_test, y_test)
        self.assertEqual(score, score2)
        rp = repr(conv)
        self.assertStartsWith(
            "SkBaseTransformStacking([Normalizer(", rp)

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_with_params(self):
        conv = SkBaseTransformStacking([LinearRegression(),
                                        DecisionTreeClassifier(max_depth=3)])
        pipe = make_pipeline(conv, DecisionTreeRegressor())
        pars = pipe.get_params(deep=True)
        self.assertIn(
            'skbasetransformstacking__models_0__model__fit_intercept', pars)
        self.assertEqual(
            pars['skbasetransformstacking__models_0__model__normalize'], True)
        conv = SkBaseTransformStacking([LinearRegression(normalize=False),
                                        DecisionTreeClassifier(max_depth=2)])
        pipe = make_pipeline(conv, DecisionTreeRegressor())
        pipe.set_params(**pars)
        pars = pipe.get_params()
        self.assertIn(
            'skbasetransformstacking__models_0__model__fit_intercept', pars)
        self.assertEqual(
            pars['skbasetransformstacking__models_0__model__normalize'], True)

    @ignore_warnings(ConvergenceWarning)
    def test_pickle(self):
        data = load_iris()
        X, y = data.data, data.target
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        conv = SkBaseTransformStacking([LinearRegression(),
                                        DecisionTreeClassifier(max_depth=3)])
        model = make_pipeline(conv, DecisionTreeRegressor())
        model.fit(X, y)

        pred = model.predict(X)

        st = BytesIO()
        pickle.dump(model, st)
        st = BytesIO(st.getvalue())
        rec = pickle.load(st)
        pred2 = rec.predict(X)
        self.assertEqualArray(pred, pred2)

    @ignore_warnings(ConvergenceWarning)
    def test_clone(self):
        conv = SkBaseTransformStacking([LinearRegression(),
                                        DecisionTreeClassifier(max_depth=3)],
                                       'predict')
        cloned = clone(conv)
        conv.test_equality(cloned, exc=True)

    @ignore_warnings(ConvergenceWarning)
    def test_grid(self):
        data = load_iris()
        X, y = data.data, data.target
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        conv = SkBaseTransformStacking([LinearRegression(),
                                        DecisionTreeClassifier(max_depth=3)])
        model = make_pipeline(conv, DecisionTreeRegressor())

        res = model.get_params(True)
        self.assertGreater(len(res), 0)

        parameters = {
            'skbasetransformstacking__models_1__model__max_depth': [2, 3]}
        clf = GridSearchCV(model, parameters)
        clf.fit(X, y)

        pred = clf.predict(X)
        self.assertEqualArray(y, pred)

    @ignore_warnings(ConvergenceWarning)
    def test_pipeline_wines(self):
        df = load_wines_dataset(shuffle=True)
        X = df.drop(['quality', 'color'], axis=1)
        y = df['quality']  # pylint: disable=E1136
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = make_pipeline(
            SkBaseTransformStacking(
                [LogisticRegression(n_jobs=1)], 'decision_function'),
            RandomForestClassifier())
        try:
            model.fit(X_train, y_train)
        except AttributeError as e:
            if compare_module_version(sklver, "0.24") < 0:
                return
            raise e
        auc_pipe = roc_auc_score(y_test == model.predict(X_test),
                                 model.predict_proba(X_test).max(axis=1))
        acc = model.score(X_test, y_test)
        accu = accuracy_score(y_test, model.predict(X_test))
        self.assertGreater(auc_pipe, 0.6)
        self.assertGreater(acc, 0.5)
        self.assertGreater(accu, 0.5)
        grid = GridSearchCV(estimator=model, param_grid={},
                            cv=3, refit='acc',
                            scoring=dict(acc=make_scorer(accuracy_score)))
        grid.fit(X, y)
        best = grid.best_estimator_
        step = grid.best_estimator_.steps[0][1]
        meth = step.method
        self.assertEqual(meth, 'decision_function')

        res = cross_val_score(model, X, y, cv=5)
        acc1 = best.score(X_test, y_test)
        accu1 = accuracy_score(y_test, best.predict(X_test))

        best.fit(X_train, y_train)
        acc2 = best.score(X_test, y_test)
        accu2 = accuracy_score(y_test, best.predict(X_test))
        self.assertGreater(res.min(), 0.5)
        self.assertGreater(min([acc2, accu2, acc1, accu1]), 0.5)


if __name__ == "__main__":
    unittest.main()
