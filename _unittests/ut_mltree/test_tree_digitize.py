# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.tree._tree import TREE_UNDEFINED
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mltree import digitize2tree


class TestTreeDigitize(ExtTestCase):

    def test_cst(self):
        self.assertEqual(TREE_UNDEFINED, -2)

    def test_tree_digitize1(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([1.0])
        expected = numpy.digitize(x, bins, right=True)
        tree = digitize2tree(bins, right=True)
        self.assertIsInstance(tree, DecisionTreeRegressor)
        pred = tree.predict(x.reshape((-1, 1)))
        self.assertEqualArray(expected, pred)

    def test_tree_digitize2(self):
        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([1.0, 2.0])
        expected = numpy.digitize(x, bins, right=True)
        tree = digitize2tree(bins, right=True)
        print("".join(export_text(tree, feature_names=['f0'])))
        print("A")
        pred = tree.predict(x.reshape((-1, 1)))
        print("B")
        self.assertEqualArray(expected, pred)

        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([0.0, 1.0, 2.5, 4.0, 10.0])
        expected = numpy.digitize(x, bins, right=True)
        tree = digitize2tree(bins, right=True)
        print("A")
        pred = tree.predict(x.reshape((-1, 1)))
        print("B")
        self.assertEqualArray(expected, pred)


if __name__ == "__main__":
    # TestTreeDigitize().test_tree_digitize2()
    unittest.main()
