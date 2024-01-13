# -*- coding: utf-8 -*-
import unittest
import numpy
from numpy.random import random
import pandas
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel import (
    run_test_sklearn_pickle,
    run_test_sklearn_clone,
    run_test_sklearn_grid_search_cv,
    DecisionTreeLogisticRegression,
)
from mlinsights.mltree import predict_leaves


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

    def test_classifier_simple_perpendicular(self):
        X = [[0.1, 0.2], [0.2, 0.3], [-0.2, -0.3], [0.4, 0.3]]
        Y = numpy.array([0, 1, 0, 1])
        dtlr = DecisionTreeLogisticRegression(
            fit_improve_algo=None, strategy="perpendicular"
        )
        self.assertRaise(lambda: dtlr.fit(X, Y), TypeError)
        X = numpy.array(X)
        Y = numpy.array(Y)
        self.assertRaise(lambda: dtlr.fit(X, Y), NotImplementedError)
        # prob = dtlr.predict_proba(X)
        # self.assertEqual(prob.shape, (4, 2))
        # dtlr.fit(X, Y, sample_weight=numpy.array([1, 1, 1, 1]))

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
        Y = X > 0.5
        X = X.reshape((100, 1))
        run_test_sklearn_pickle(lambda: LogisticRegression(), X, Y)
        run_test_sklearn_pickle(
            lambda: DecisionTreeLogisticRegression(fit_improve_algo=None), X, Y
        )

    def test_classifier_clone(self):
        run_test_sklearn_clone(
            lambda: DecisionTreeLogisticRegression(fit_improve_algo=None)
        )

    def test_classifier_grid_search(self):
        X = random(100)
        Y = X > 0.5
        X = X.reshape((100, 1))
        self.assertRaise(
            lambda: run_test_sklearn_grid_search_cv(
                lambda: DecisionTreeLogisticRegression(fit_improve_algo=None), X, Y
            ),
            AssertionError,
        )
        res = run_test_sklearn_grid_search_cv(
            lambda: DecisionTreeLogisticRegression(fit_improve_algo=None),
            X,
            Y,
            max_depth=[2, 3],
        )
        self.assertIn("model", res)
        self.assertIn("score", res)
        self.assertGreater(res["score"], 0)
        self.assertLesser(res["score"], 1)

    def test_iris(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
        dtlr = DecisionTreeLogisticRegression(fit_improve_algo=None)
        self.assertRaise(lambda: dtlr.fit(X_train, y_train), AssertionError)
        y = y % 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
        self.assertRaise(
            lambda: DecisionTreeLogisticRegression(fit_improve_algo="fit_improve_algo"),
            ValueError,
        )
        dtlr = DecisionTreeLogisticRegression(fit_improve_algo="intercept_sort_always")
        self.assertRaise(lambda: dtlr.fit(X_train, y_train), AssertionError)
        y = y % 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
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

    def test_decision_path(self):
        data = load_iris()
        X, y = data.data, data.target
        y = y % 2
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        dtlr = DecisionTreeLogisticRegression()
        dtlr.fit(X_train, y_train)
        path = dtlr.decision_path(X_test)
        self.assertEqual(path.shape[0], X_test.shape[0])
        self.assertGreater(path.shape[1], X_test.shape[1])
        indices = dtlr.get_leaves_index()
        self.assertGreater(indices.shape[0], 3)
        leaves = predict_leaves(dtlr, X_test)
        self.assertEqual(leaves.shape[0], X_test.shape[0])

    def test_classifier_strat(self):
        X = numpy.array([[0.1, 0.2], [0.2, 0.3], [-0.2, -0.3], [0.4, 0.3]])
        Y = numpy.array([0, 1, 0, 1])
        dtlr = DecisionTreeLogisticRegression(fit_improve_algo=None, strategy="")
        self.assertRaise(lambda: dtlr.fit(X, Y), ValueError)


if __name__ == "__main__":
    unittest.main()
