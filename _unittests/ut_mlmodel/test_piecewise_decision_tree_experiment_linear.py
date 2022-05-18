# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
import numpy
from sklearn.tree._criterion import MSE  # pylint: disable=E0611
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel.piecewise_tree_regression import PiecewiseTreeRegressor
from mlinsights.mlmodel._piecewise_tree_regression_common import (  # pylint: disable=E0611,E0401
    _test_criterion_init, _test_criterion_node_impurity,
    _test_criterion_node_impurity_children, _test_criterion_update,
    _test_criterion_node_value, _test_criterion_proxy_impurity_improvement,
    _test_criterion_impurity_improvement
)
from mlinsights.mlmodel.piecewise_tree_regression_criterion_linear import LinearRegressorCriterion  # pylint: disable=E0611, E0401


class TestPiecewiseDecisionTreeExperimentLinear(ExtTestCase):

    def test_criterions(self):
        X = numpy.array([[10., 12., 13.]]).T
        y = numpy.array([20., 22., 23.])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        self.assertNotEmpty(c1)
        self.assertNotEmpty(c2)
        w = numpy.ones((y.shape[0],))
        self.assertEqual(w.sum(), X.shape[0])
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1., ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1., ind, 0, y.shape[0])
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pyx#L886
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertGreater(i1, i2)
        self.assertGreater(i2, 0)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1., 2., 3.]]).T
        y = numpy.array([1., 2., 3.])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1., ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1., ind, 0, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertGreater(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1., 2., 10., 11.]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1., ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1., ind, 0, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertGreater(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1., 2., 10., 11.]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.array([0, 3, 2, 1], dtype=ind.dtype)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1., ind, 1, y.shape[0])
        _test_criterion_init(c2, ys, w, 1., ind, 1, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertGreater(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        for i in range(2, 4):
            _test_criterion_update(c1, i)
            _test_criterion_update(c2, i)
            left1, right1 = _test_criterion_node_impurity_children(c1)
            left2, right2 = _test_criterion_node_impurity_children(c2)
            self.assertGreater(left1, left2)
            self.assertGreater(right1, right2)
            v1 = _test_criterion_node_value(c1)
            v2 = _test_criterion_node_value(c2)
            self.assertEqual(v1, v2)
            try:
                # scikit-learn >= 0.24
                p1 = _test_criterion_impurity_improvement(
                    c1, 0., left1, right1)
                p2 = _test_criterion_impurity_improvement(
                    c2, 0., left2, right2)
            except TypeError:
                # scikit-learn < 0.23
                p1 = _test_criterion_impurity_improvement(c1, 0.)
                p2 = _test_criterion_impurity_improvement(c2, 0.)
            self.assertGreater(p1, p2 - 1.)

            dest = numpy.empty((2, ))
            c2.node_beta(dest)
            self.assertGreater(dest[0], 0)
            self.assertGreater(dest[1], 0)

    def test_criterions_check_value(self):
        X = numpy.array([[10., 12., 13.]]).T
        y = numpy.array([[20., 22., 23.]]).T
        c2 = LinearRegressorCriterion.create(X, y)
        coef = numpy.empty((3, ))
        c2.node_beta(coef)
        self.assertEqual(coef[:2], numpy.array([1, 10]))

    def test_decision_tree_criterion(self):
        X = numpy.array([[1., 2., 10., 11.]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        clr1 = DecisionTreeRegressor(max_depth=1)
        clr1.fit(X, y)
        p1 = clr1.predict(X)

        crit = LinearRegressorCriterion(1, X)
        clr2 = DecisionTreeRegressor(criterion=crit, max_depth=1)
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1, p2)
        self.assertEqual(clr1.tree_.node_count, clr2.tree_.node_count)

    def test_decision_tree_criterion_iris(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr1 = DecisionTreeRegressor()
        clr1.fit(X, y)
        p1 = clr1.predict(X)
        clr2 = DecisionTreeRegressor(criterion=LinearRegressorCriterion(1, X))
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1.shape, p2.shape)

    def test_decision_tree_criterion_iris_dtc(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr1 = DecisionTreeRegressor()
        clr1.fit(X, y)
        p1 = clr1.predict(X)
        clr2 = PiecewiseTreeRegressor(criterion='mselin')
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1.shape, p2.shape)
        self.assertTrue(hasattr(clr2, 'betas_'))
        self.assertTrue(hasattr(clr2, 'leaves_mapping_'))
        self.assertEqual(len(clr2.leaves_index_), clr2.tree_.n_leaves)
        self.assertEqual(len(clr2.leaves_mapping_), clr2.tree_.n_leaves)
        self.assertEqual(clr2.betas_.shape[1], X.shape[1] + 1)
        self.assertEqual(clr2.betas_.shape[0], clr2.tree_.n_leaves)
        sc1 = clr1.score(X, y)
        sc2 = clr2.score(X, y)
        self.assertGreater(sc1, sc2)
        mp = clr2._mapping_train(X)  # pylint: disable=W0212
        self.assertIsInstance(mp, dict)
        self.assertGreater(len(mp), 2)

    def test_decision_tree_criterion_iris_dtc_traintest(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr1 = DecisionTreeRegressor()
        clr1.fit(X_train, y_train)
        p1 = clr1.predict(X_train)
        clr2 = PiecewiseTreeRegressor(criterion='mselin')
        clr2.fit(X_train, y_train)
        p2 = clr2.predict(X_train)
        self.assertEqual(p1.shape, p2.shape)
        self.assertTrue(hasattr(clr2, 'betas_'))
        self.assertTrue(hasattr(clr2, 'leaves_mapping_'))
        self.assertEqual(len(clr2.leaves_index_), clr2.tree_.n_leaves)
        self.assertEqual(len(clr2.leaves_mapping_), clr2.tree_.n_leaves)
        self.assertEqual(clr2.betas_.shape[1], X.shape[1] + 1)
        self.assertEqual(clr2.betas_.shape[0], clr2.tree_.n_leaves)
        sc1 = clr1.score(X_test, y_test)
        sc2 = clr2.score(X_test, y_test)
        self.assertGreater(abs(sc1 - sc2), -0.1)


if __name__ == "__main__":
    unittest.main()
