# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from numpy.random import random
import pandas
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel import (
    test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv,
    DecisionTreeLogisticRegression
)


class TestDecisionTreeLogisticRegression(ExtTestCase):

    def test_classifier_simple(self):
        X = [[0.1, 0.2], [0.2, 0.3], [-0.2, -0.3], [0.4, 0.3]]
        Y = numpy.array([0, 1, 0, 1])
        dtlr = DecisionTreeLogisticRegression(fit_improve_algo=None)
        self.assertRaise(lambda: dtlr.fit(X, Y), TypeError)
        X = numpy.array(X)
        Y = numpy.array(Y)
        dtlr.fit(X, Y)
        prob = dtlr.predict_proba(X)
        self.assertEqual(prob.shape, (4, 2))
        dtlr.fit(X, Y, sample_weight=numpy.array([1, 1, 1, 1]))

    def test_pandas(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [-0.2, 0.3]]))
        Y = numpy.array([0, 1])
        clq = DecisionTreeLogisticRegression(fit_improve_algo=None)
        clq.fit(X, Y)
        pred2 = clq.predict(X)
        self.assertEqual(numpy.array([0, 1]), pred2)

    def test_classifier_list(self):
        X = [[0.1, 0.2], [0.2, 0.3]]
        Y = numpy.array([0, 1])
        clq = DecisionTreeLogisticRegression(fit_improve_algo=None)
        self.assertRaise(lambda: clq.fit(X, Y), TypeError)

    def test_classifier_pickle(self):
        X = random(100)
        Y = X > 0.5  # pylint: disable=W0143
        X = X.reshape((100, 1))  # pylint: disable=E1101
        test_sklearn_pickle(lambda: LogisticRegression(), X, Y)
        test_sklearn_pickle(lambda: DecisionTreeLogisticRegression(
            fit_improve_algo=None), X, Y)

    def test_classifier_clone(self):
        test_sklearn_clone(
            lambda: DecisionTreeLogisticRegression(fit_improve_algo=None))

    def test_classifier_grid_search(self):
        X = random(100)
        Y = X > 0.5  # pylint: disable=W0143
        X = X.reshape((100, 1))  # pylint: disable=E1101
        self.assertRaise(lambda: test_sklearn_grid_search_cv(
            lambda: DecisionTreeLogisticRegression(fit_improve_algo=None), X, Y), ValueError)
        res = test_sklearn_grid_search_cv(lambda: DecisionTreeLogisticRegression(fit_improve_algo=None),
                                          X, Y, max_depth=[2, 3])
        self.assertIn('model', res)
        self.assertIn('score', res)
        self.assertGreater(res['score'], 0)
        self.assertLesser(res['score'], 1)

    def test_iris(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=11)
        dtlr = DecisionTreeLogisticRegression(fit_improve_algo=None)
        self.assertRaise(lambda: dtlr.fit(X_train, y_train), RuntimeError)
        y = y % 2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=11)
        dtlr.fit(X_train, y_train)
        depth = dtlr.tree_depth_
        self.assertGreater(depth, 2)
        sc = dtlr.score(X_test, y_test)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        sc2 = lr.score(X_test, y_test)
        self.assertGreater(sc, sc2)
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        sc3 = dt.score(X_test, y_test)
        self.assertGreater(sc3, sc2)
        sc = dtlr.score(X_train, y_train)
        sc2 = lr.score(X_train, y_train)
        sc3 = dt.score(X_train, y_train)
        self.assertGreater(sc, sc2)
        self.assertGreater(sc3, sc2)

    def test_iris_fit_improve(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=11)
        self.assertRaise(lambda: DecisionTreeLogisticRegression(
            fit_improve_algo='fit_improve_algo'), ValueError)
        dtlr = DecisionTreeLogisticRegression(
            fit_improve_algo='intercept_sort_always')
        self.assertRaise(lambda: dtlr.fit(X_train, y_train), RuntimeError)
        y = y % 2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=11)
        dtlr.fit(X_train, y_train)
        depth = dtlr.tree_depth_
        self.assertGreater(depth, 2)
        sc = dtlr.score(X_test, y_test)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        sc2 = lr.score(X_test, y_test)
        self.assertGreater(sc, sc2)
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        sc3 = dt.score(X_test, y_test)
        self.assertGreater(sc3, sc2)
        sc = dtlr.score(X_train, y_train)
        sc2 = lr.score(X_train, y_train)
        sc3 = dt.score(X_train, y_train)
        self.assertGreater(sc, sc2)
        self.assertGreater(sc3, sc2)


if __name__ == "__main__":
    unittest.main()
