# -*- coding: utf-8 -*-
import unittest
import numpy
from sklearn.tree._criterion import MSE
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel.piecewise_tree_regression import PiecewiseTreeRegressor
from mlinsights.mlmodel._piecewise_tree_regression_common import (
    _test_criterion_init,
    _test_criterion_node_impurity,
    _test_criterion_node_impurity_children,
    _test_criterion_update,
    _test_criterion_node_value,
    _test_criterion_proxy_impurity_improvement,
    _test_criterion_impurity_improvement,
)
from mlinsights.mlmodel._piecewise_tree_regression_common import (
    _test_criterion_check,
    assert_criterion_equal,
)
from mlinsights.mlmodel.piecewise_tree_regression_criterion import (
    SimpleRegressorCriterion,
)


class TestPiecewiseDecisionTreeExperiment(ExtTestCase):
    @unittest.skip(
        reason="self.y = y raises: Fatal Python error: "
        "__pyx_fatalerror: Acquisition count is"
    )
    def test_criterions(self):
        X = numpy.array([[1.0, 2.0]]).T
        y = numpy.array([1.0, 2.0])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterion(1, X.shape[0])
        self.assertNotEmpty(c1)
        self.assertNotEmpty(c2)
        w = numpy.ones((y.shape[0],))
        self.assertEqual(w.sum(), X.shape[0])
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 0, y.shape[0])
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx#L886
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertEqual(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1.0, 2.0, 3.0]]).T
        y = numpy.array([1.0, 2.0, 3.0])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterion(1, X.shape[0])
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 0, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertAlmostEqual(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1.0, 2.0, 10.0, 11.0]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterion(1, X.shape[0])
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 0, y.shape[0])
        _test_criterion_check(c1)
        _test_criterion_check(c2)
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        _test_criterion_check(c1)
        _test_criterion_check(c2)
        assert_criterion_equal(c1, c2)
        self.assertAlmostEqual(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        _test_criterion_check(c2)
        assert_criterion_equal(c1, c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        _test_criterion_check(c2)
        assert_criterion_equal(c1, c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        for i in range(1, 4):
            _test_criterion_check(c2)
            _test_criterion_update(c1, i)
            _test_criterion_update(c2, i)
            _test_criterion_check(c2)
            assert_criterion_equal(c1, c2)
            left1, right1 = _test_criterion_node_impurity_children(c1)
            left2, right2 = _test_criterion_node_impurity_children(c2)
            self.assertAlmostEqual(left1, left2)
            self.assertAlmostEqual(right1, right2)
            v1 = _test_criterion_node_value(c1)
            v2 = _test_criterion_node_value(c2)
            self.assertEqual(v1, v2)
            try:
                # scikit-learn >= 0.24
                p1 = _test_criterion_impurity_improvement(c1, 0.0, left1, right1)
                p2 = _test_criterion_impurity_improvement(c2, 0.0, left2, right2)
            except TypeError:
                # scikit-learn < 0.24
                p1 = _test_criterion_impurity_improvement(c1, 0.0)
                p2 = _test_criterion_impurity_improvement(c2, 0.0)
            self.assertAlmostEqual(p1, p2)

        X = numpy.array([[1.0, 2.0, 10.0, 11.0]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterion(1, X.shape[0])
        w = numpy.ones((y.shape[0],))
        ind = numpy.array([0, 3, 2, 1], dtype=ind.dtype)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 1, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 1, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertAlmostEqual(i1, i2)
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
            self.assertAlmostEqual(left1, left2)
            self.assertAlmostEqual(right1, right2)
            v1 = _test_criterion_node_value(c1)
            v2 = _test_criterion_node_value(c2)
            self.assertEqual(v1, v2)
            try:
                # scikit-learn >= 0.24
                p1 = _test_criterion_impurity_improvement(c1, 0.0, left1, right1)
                p2 = _test_criterion_impurity_improvement(c2, 0.0, left2, right2)
            except TypeError:
                # scikit-learn < 0.24
                p1 = _test_criterion_impurity_improvement(c1, 0.0)
                p2 = _test_criterion_impurity_improvement(c2, 0.0)
            self.assertAlmostEqual(p1, p2)

    def test_decision_tree_criterion(self):
        X = numpy.array([[1.0, 2.0, 10.0, 11.0]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        clr1 = DecisionTreeRegressor(max_depth=1)
        clr1.fit(X, y)
        p1 = clr1.predict(X)

        crit = SimpleRegressorCriterion(
            1 if len(y.shape) <= 1 else y.shape[1], X.shape[0]
        )
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
        clr2 = DecisionTreeRegressor(
            criterion=SimpleRegressorCriterion(
                1 if len(y.shape) <= 1 else y.shape[1], X.shape[0]
            )
        )
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1[:10], p2[:10])

    def test_decision_tree_criterion_iris_dtc(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr1 = DecisionTreeRegressor()
        clr1.fit(X, y)
        p1 = clr1.predict(X)
        clr2 = PiecewiseTreeRegressor(criterion="simple")
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1[:10], p2[:10])


if __name__ == "__main__":
    unittest.main(verbosity=2)
