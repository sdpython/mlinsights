# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
import numpy
from scipy.spatial.distance import cdist
from sklearn import datasets
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel import KMeansL1L2
from mlinsights.mlmodel._kmeans_022 import _assign_labels_array


class TestKMeansL1L2(ExtTestCase):

    def test_kmeans_l2(self):
        iris = datasets.load_iris()
        X = iris.data
        clr = KMeansL1L2(4)
        clr.fit(X)
        cls = set(clr.predict(X))
        self.assertEqual({0, 1, 2, 3}, cls)

    def test_kmeans_l2_small(self):
        iris = datasets.load_iris()
        X = iris.data
        X = X[:6]
        clr = KMeansL1L2(4, n_jobs=1)
        clr.fit(X)
        cls = set(clr.predict(X))
        self.assertEqual({0, 1, 2, 3}, cls)

    def test_kmeans_l1_small(self):
        iris = datasets.load_iris()
        X = iris.data
        X = X[:6]
        clr = KMeansL1L2(4, norm='L1', n_jobs=1)
        clr.fit(X)
        cls = set(clr.predict(X))
        self.assertEqual({0, 1, 2, 3}, cls)

    def test_kmeans_l1_iris(self):
        iris = datasets.load_iris()
        X = iris.data
        clr = KMeansL1L2(4, norm='L1')
        clr.fit(X)
        cls = set(clr.predict(X))
        self.assertEqual({0, 1, 2, 3}, cls)

    def test_kmeans_l2_iris(self):
        iris = datasets.load_iris()
        X = iris.data
        clr = KMeansL1L2(4, norm='L2')
        clr.fit(X)
        cls = set(clr.predict(X))
        self.assertEqual({0, 1, 2, 3}, cls)

    def test_kmeans_l1_check(self):
        X = numpy.array([[-10, 1, 2, 3, 4, 10],
                         [-10, 1, 2, 3, 4, 10]]).T
        clr = KMeansL1L2(2, norm='L1')
        clr.fit(X)
        cls = set(clr.predict(X))
        self.assertEqual({0, 1}, cls)
        self.assertEqual(clr.cluster_centers_.shape, (2, 2))
        self.assertEqualArray(clr.cluster_centers_.max(), [3, 3])
        tr = clr.transform(X)
        self.assertEqual(tr.shape, (X.shape[0], 2))
        tr = clr.transform([[3, 3]])
        self.assertEqualArray(tr.min(), [0])

    def test__assign_labels_array(self):
        X = numpy.array([[1., 2.], [3.5, 4.]])
        sample_weight = numpy.array([1., 1.1])
        centers = X.copy()
        labels = numpy.array([0, 1])
        x_squared_norms = numpy.array([5., 3.1])
        distances = cdist(X, centers)
        res = _assign_labels_array(
            X, sample_weight, x_squared_norms, centers,
            labels, distances)
        self.assertIsInstance(res, numpy.float64)


if __name__ == "__main__":
    unittest.main()
