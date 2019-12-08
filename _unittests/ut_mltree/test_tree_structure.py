# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mltree import tree_leave_index, tree_node_range, tree_leave_neighbors
from mlinsights.mltree.tree_structure import tree_find_common_node


class TestTreeStructure(ExtTestCase):

    def test_iris(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr = DecisionTreeClassifier(max_depth=3)
        clr.fit(X, y)
        leaves = tree_leave_index(clr)
        self.assertNotEmpty(leaves)

    def test_cube(self):
        X = numpy.array([[0, 0], [0, 1], [0, 2],
                         [1, 0], [1, 1], [1, 2],
                         [2, 0], [2, 1], [2, 2]])
        y = list(range(X.shape[0]))
        clr = DecisionTreeClassifier(max_depth=4)
        clr.fit(X, y)
        leaves = tree_leave_index(clr)
        exp = {
            8: numpy.array([[1.5, numpy.nan], [1.5, numpy.nan]]),
            4: numpy.array([[0.5, 1.5], [0.5, 1.5]])
        }
        for le in leaves:
            ra = tree_node_range(clr, le)
            cl = clr.tree_.value[le]  # pylint: disable=E1136
            am = numpy.argmax(cl.ravel())
            if am in exp:
                self.assertEqualArray(ra, exp[am])
        self.assertNotEmpty(leaves)
        common = tree_find_common_node(clr, 0, 1)
        self.assertEqual(common, (0, [], [1]))

    def test_tree_leave_neighbors(self):
        X = numpy.array([[0, 0], [0, 1], [0, 2],
                         [1, 0], [1, 1], [1, 2],
                         [2, 0], [2, 1], [2, 2]])
        y = list(range(X.shape[0]))
        clr = DecisionTreeClassifier(max_depth=4)
        clr.fit(X, y)
        nei = tree_leave_neighbors(clr)
        self.assertEqual(len(nei), 12)
        self.assertIsInstance(nei, dict)
        for k, v in nei.items():
            self.assertIsInstance(k, tuple)
            self.assertIsInstance(v, list)
            self.assertEqual(len(k), 2)
            self.assertEqual(len(v), 1)
            self.assertIn(v[0][0], (0, 1))
            self.assertIsInstance(v[0][1], tuple)
            self.assertIsInstance(v[0][2], tuple)
            self.assertEqual(len(v[0][1]), 2)
            self.assertEqual(len(v[0][2]), 2)

    def test_tree_leave_neighbors2(self):
        X = numpy.array([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                         [1, 0, 0], [1, 0, 1], [1, 0, 2],
                         [2, 0, 0], [2, 0, 1], [2, 0, 2]])
        y = list(range(X.shape[0]))
        clr = DecisionTreeClassifier(max_depth=4)
        clr.fit(X, y)
        nei = tree_leave_neighbors(clr)
        self.assertEqual(len(nei), 12)
        self.assertIsInstance(nei, dict)
        for k, v in nei.items():
            self.assertIsInstance(k, tuple)
            self.assertIsInstance(v, list)
            self.assertEqual(len(k), 2)
            self.assertEqual(len(v), 1)
            self.assertIn(v[0][0], (0, 2))
            self.assertIsInstance(v[0][1], tuple)
            self.assertIsInstance(v[0][2], tuple)
            self.assertEqual(len(v[0][1]), 3)
            self.assertEqual(len(v[0][2]), 3)


if __name__ == "__main__":
    unittest.main()
