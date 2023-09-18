# -*- coding: utf-8 -*-
import unittest
import numpy
from sklearn.tree import DecisionTreeRegressor

try:
    from sklearn.tree._tree import TREE_UNDEFINED
except ImportError:
    TREE_UNDEFINED = None
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mltree import digitize2tree


class TestTreeDigitize(ExtTestCase):
    @unittest.skipIf(TREE_UNDEFINED is None, reason="nothing to test")
    def test_cst(self):
        self.assertEqual(TREE_UNDEFINED, -2)

    def test_exc(self):
        bins = numpy.array([0.0, 1.0])
        self.assertRaise(lambda: digitize2tree(bins, right=False), RuntimeError)
        bins = numpy.array([1.0, 0.0])
        self.assertRaise(lambda: digitize2tree(bins, right=False), RuntimeError)

    def test_tree_digitize1(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([1.0])
        expected = numpy.digitize(x, bins, right=True).astype(numpy.float64)
        tree = digitize2tree(bins, right=True)
        self.assertIsInstance(tree, DecisionTreeRegressor)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)
        expected = numpy.digitize(bins, bins, right=True).astype(numpy.float64)
        pred = tree.predict(bins.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)

    def test_tree_digitize2(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([1.0, 2.0])
        expected = numpy.digitize(x, bins, right=True).astype(numpy.float64)
        tree = digitize2tree(bins, right=True)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)
        expected = numpy.digitize(bins, bins, right=True).astype(numpy.float64)
        pred = tree.predict(bins.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)

    def test_tree_digitize3(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([1.0, 2.0, 3.5])
        expected = numpy.digitize(x, bins, right=True).astype(numpy.float64)
        tree = digitize2tree(bins, right=True)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)
        expected = numpy.digitize(bins, bins, right=True).astype(numpy.float64)
        pred = tree.predict(bins.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)

    def test_tree_digitize4(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([0.0, 1.0, 2.5, 4.0])
        expected = numpy.digitize(x, bins, right=True).astype(numpy.float64)
        tree = digitize2tree(bins, right=True)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)
        expected = numpy.digitize(bins, bins, right=True).astype(numpy.float64)
        pred = tree.predict(bins.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)

    def test_tree_digitize5(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([0.0, 1.0, 2.5, 4.0, 7.0])
        expected = numpy.digitize(x, bins, right=True).astype(numpy.float64)
        tree = digitize2tree(bins, right=True)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)
        expected = numpy.digitize(bins, bins, right=True).astype(numpy.float64)
        pred = tree.predict(bins.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)

    def test_tree_digitize5_false(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([0.0, 1.0, 2.5, 4.0, 7.0])
        bins[:] = bins[::-1].copy()
        expected = numpy.digitize(x, bins, right=True).astype(numpy.float64)
        tree = digitize2tree(bins, right=True)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)
        expected = numpy.digitize(bins, bins, right=True).astype(numpy.float64)
        pred = tree.predict(bins.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)

    def test_tree_digitize_bigger(self):
        x = numpy.array([0, 1, 2, 3, 4, 5, 6, -1], dtype=numpy.float32)
        bins = numpy.array([0, 1, 2, 3, 4], dtype=numpy.float32)
        expected = numpy.digitize(x, bins, right=True).astype(numpy.float64)
        tree = digitize2tree(bins, right=True)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)
        expected = numpy.digitize(bins, bins, right=True).astype(numpy.float64)
        pred = tree.predict(bins.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)


if __name__ == "__main__":
    unittest.main()
