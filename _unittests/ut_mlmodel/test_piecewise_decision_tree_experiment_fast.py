# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
import numpy
import sklearn
from sklearn.tree._criterion import MSE  # pylint: disable=E0611
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper import compare_module_version
from mlinsights.mlmodel.piecewise_tree_regression import PiecewiseTreeRegressor
from mlinsights.mlmodel._piecewise_tree_regression_common import (  # pylint: disable=E0611, E0401
    _test_criterion_init, _test_criterion_node_impurity,
    _test_criterion_node_impurity_children, _test_criterion_update,
    _test_criterion_node_value, _test_criterion_proxy_impurity_improvement,
    _test_criterion_impurity_improvement
)
from mlinsights.mlmodel.piecewise_tree_regression_criterion import SimpleRegressorCriterion  # pylint: disable=E0611, E0401
from mlinsights.mlmodel.piecewise_tree_regression_criterion_fast import SimpleRegressorCriterionFast  # pylint: disable=E0611, E0401


class TestPiecewiseDecisionTreeExperimentFast(ExtTestCase):

    def test_criterions(self):
        X = numpy.array([[1., 2.]]).T
        y = numpy.array([1., 2.])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterionFast(X)
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
        self.assertEqual(i1, i2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1., 2., 3.]]).T
        y = numpy.array([1., 2., 3.])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterionFast(X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1., ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1., ind, 0, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertAlmostEqual(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1., 2., 10., 11.]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterionFast(X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1., ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1., ind, 0, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertAlmostEqual(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        for i in range(1, 4):
            _test_criterion_update(c1, i)
            _test_criterion_update(c2, i)
            left1, right1 = _test_criterion_node_impurity_children(c1)
            left2, right2 = _test_criterion_node_impurity_children(c2)
            self.assertAlmostEqual(left1, left2)
            self.assertAlmostEqual(right1, right2)
            v1 = _test_criterion_node_value(c1)
            v2 = _test_criterion_node_value(c2)
            self.assertEqual(v1, v2)
            p1 = _test_criterion_impurity_improvement(c1, 0.)
            p2 = _test_criterion_impurity_improvement(c2, 0.)
            self.assertAlmostEqual(p1, p2)

        X = numpy.array([[1., 2., 10., 11.]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = SimpleRegressorCriterionFast(X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.array([0, 3, 2, 1], dtype=ind.dtype)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1., ind, 1, y.shape[0])
        _test_criterion_init(c2, ys, w, 1., ind, 1, y.shape[0])
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
            p1 = _test_criterion_impurity_improvement(c1, 0.)
            p2 = _test_criterion_impurity_improvement(c2, 0.)
            self.assertAlmostEqual(p1, p2)

    @unittest.skipIf(compare_module_version(sklearn.__version__, "0.21") < 0,
                     reason="Only implemented for Criterion API from sklearn >= 0.21")
    def test_decision_tree_criterion(self):
        X = numpy.array([[1., 2., 10., 11.]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        clr1 = DecisionTreeRegressor(max_depth=1)
        clr1.fit(X, y)
        p1 = clr1.predict(X)

        crit = SimpleRegressorCriterionFast(X)
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
        clr2 = DecisionTreeRegressor(criterion=SimpleRegressorCriterionFast(X))
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1[:10], p2[:10])

    def test_decision_tree_criterion_iris_dtc(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr1 = DecisionTreeRegressor()
        clr1.fit(X, y)
        p1 = clr1.predict(X)
        clr2 = PiecewiseTreeRegressor(criterion='simple')
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1[:10], p2[:10])


if __name__ == "__main__":
    unittest.main()
