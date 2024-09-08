import unittest
import warnings
import numpy
import packaging.version as pv
from sklearn.tree._criterion import MSE
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, __version__ as skl_ver
from sklearn.model_selection import train_test_split
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel.piecewise_tree_regression import PiecewiseTreeRegressor

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from mlinsights.mlmodel._piecewise_tree_regression_common import (
        _test_criterion_init,
        _test_criterion_node_impurity,
        _test_criterion_node_impurity_children,
        _test_criterion_update,
        _test_criterion_node_value,
        _test_criterion_proxy_impurity_improvement,
        _test_criterion_impurity_improvement,
    )
    from mlinsights.mlmodel.piecewise_tree_regression_criterion_linear import (
        LinearRegressorCriterion,
    )


class TestPiecewiseDecisionTreeExperimentLinear(ExtTestCase):
    @unittest.skipIf(
        pv.Version(skl_ver) < pv.Version("1.3.3"),
        reason="it works with the main branch and the same cython",
    )
    def test_criterions(self):
        X = numpy.array([[10.0, 12.0, 13.0]]).T
        y = numpy.array([20.0, 22.0, 23.0])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        self.assertNotEmpty(c1)
        self.assertNotEmpty(c2)
        w = numpy.ones((y.shape[0],))
        self.assertEqual(w.sum(), X.shape[0])
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 0, y.shape[0])
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx#L886
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

        X = numpy.array([[1.0, 2.0, 3.0]]).T
        y = numpy.array([1.0, 2.0, 3.0])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 0, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertGreater(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1.0, 2.0, 10.0, 11.0]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.arange(y.shape[0]).astype(numpy.int64)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 0, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 0, y.shape[0])
        i1 = _test_criterion_node_impurity(c1)
        i2 = _test_criterion_node_impurity(c2)
        self.assertGreater(i1, i2)
        v1 = _test_criterion_node_value(c1)
        v2 = _test_criterion_node_value(c2)
        self.assertEqual(v1, v2)
        p1 = _test_criterion_proxy_impurity_improvement(c1)
        p2 = _test_criterion_proxy_impurity_improvement(c2)
        self.assertTrue(numpy.isnan(p1), numpy.isnan(p2))

        X = numpy.array([[1.0, 2.0, 10.0, 11.0]]).T
        y = numpy.array([0.9, 1.1, 1.9, 2.1])
        c1 = MSE(1, X.shape[0])
        c2 = LinearRegressorCriterion(1, X)
        w = numpy.ones((y.shape[0],))
        ind = numpy.array([0, 3, 2, 1], dtype=ind.dtype)
        ys = y.astype(float).reshape((y.shape[0], 1))
        _test_criterion_init(c1, ys, w, 1.0, ind, 1, y.shape[0])
        _test_criterion_init(c2, ys, w, 1.0, ind, 1, y.shape[0])
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
            p1 = _test_criterion_impurity_improvement(c1, 0.0, left1, right1)
            p2 = _test_criterion_impurity_improvement(c2, 0.0, left2, right2)
            self.assertGreater(p1, p2 - 1.0)

            dest = numpy.empty((2,))
            c2.node_beta(dest)
            self.assertGreater(dest[0], 0)
            self.assertGreater(dest[1], 0)

    @unittest.skipIf(
        pv.Version(skl_ver) < pv.Version("1.3.3"),
        reason="it works with the main branch and the same cython",
    )
    def test_criterions_check_value(self):
        X = numpy.array([[10.0, 12.0, 13.0]]).T
        y = numpy.array([[20.0, 22.0, 23.0]]).T
        c2 = LinearRegressorCriterion.create(X, y)
        coef = numpy.empty((3,), dtype=X.dtype)
        c2.node_beta(coef)
        self.assertEqualArray(coef[:2], numpy.array([1, 10], dtype=X.dtype), atol=1e-8)

    @unittest.skipIf(
        pv.Version(skl_ver) < pv.Version("1.3.3"),
        reason="it works with the main branch and the same cython",
    )
    def test_decision_tree_criterion_linear(self):
        X = numpy.array([[1.0, 2.0, 10.0, 11.0]]).T
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

    @unittest.skipIf(
        pv.Version(skl_ver) < pv.Version("1.3.3"),
        reason="it works with the main branch and the same cython",
    )
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

    @unittest.skipIf(
        pv.Version(skl_ver) < pv.Version("1.3.3"),
        reason="it works with the main branch and the same cython",
    )
    def test_decision_tree_criterion_iris_dtc(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr1 = DecisionTreeRegressor()
        clr1.fit(X, y)
        p1 = clr1.predict(X)
        clr2 = PiecewiseTreeRegressor(criterion="mselin")
        clr2.fit(X, y)
        p2 = clr2.predict(X)
        self.assertEqual(p1.shape, p2.shape)
        self.assertTrue(hasattr(clr2, "betas_"))
        self.assertTrue(hasattr(clr2, "leaves_mapping_"))
        self.assertEqual(len(clr2.leaves_index_), clr2.tree_.n_leaves)
        self.assertEqual(len(clr2.leaves_mapping_), clr2.tree_.n_leaves)
        self.assertEqual(clr2.betas_.shape[1], X.shape[1] + 1)
        self.assertEqual(clr2.betas_.shape[0], clr2.tree_.n_leaves)
        sc1 = clr1.score(X, y)
        sc2 = clr2.score(X, y)
        self.assertGreater(sc1, sc2)
        mp = clr2._mapping_train(X)
        self.assertIsInstance(mp, dict)
        self.assertGreater(len(mp), 2)

    @unittest.skipIf(
        pv.Version(skl_ver) < pv.Version("1.3.3"),
        reason="it works with the main branch and the same cython",
    )
    def test_decision_tree_criterion_iris_dtc_traintest(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr1 = DecisionTreeRegressor()
        clr1.fit(X_train, y_train)
        p1 = clr1.predict(X_train)
        clr2 = PiecewiseTreeRegressor(criterion="mselin")
        clr2.fit(X_train, y_train)
        p2 = clr2.predict(X_train)
        self.assertEqual(p1.shape, p2.shape)
        self.assertTrue(hasattr(clr2, "betas_"))
        self.assertTrue(hasattr(clr2, "leaves_mapping_"))
        self.assertEqual(len(clr2.leaves_index_), clr2.tree_.n_leaves)
        self.assertEqual(len(clr2.leaves_mapping_), clr2.tree_.n_leaves)
        self.assertEqual(clr2.betas_.shape[1], X.shape[1] + 1)
        self.assertEqual(clr2.betas_.shape[0], clr2.tree_.n_leaves)
        sc1 = clr1.score(X_test, y_test)
        sc2 = clr2.score(X_test, y_test)
        self.assertGreater(abs(sc1 - sc2), -0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
