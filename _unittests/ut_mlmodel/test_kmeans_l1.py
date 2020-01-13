# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from numpy.random import RandomState
from sklearn import datasets
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel import KMeansL1L2


class TestKMeansL1L2(ExtTestCase):

    def test_kmeans_l2(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clr = KMeansL1L2(4)
        clr.fit(X, y)
        cls = set(clr.predict(X, y))
        self.assertEqual({0, 1, 2, 3}, cls)


if __name__ == "__main__":
    unittest.main()
