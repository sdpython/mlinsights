# -*- coding: utf-8 -*-
import unittest
import numpy as np
from scipy import sparse as sp
from sklearn.utils._testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_almost_equal,
    assert_raise_message,
)
from sklearn.metrics.cluster import v_measure_score
from sklearn.datasets import make_blobs
from mlinsights.ext_test_case import ExtTestCase, ignore_warnings
from mlinsights.mlmodel import KMeansL1L2


class TestKMeansL1L2Sklearn(ExtTestCase):
    # non centered, sparse centers to check the
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    n_clusters, n_features = centers.shape
    X, true_labels = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )[:2]
    X_csr = sp.csr_matrix(X)

    def do_test_kmeans_results(self, representation, algo, dtype, norm, sw):
        # cheks that kmeans works as intended
        array_constr = {"dense": np.array, "sparse": sp.csr_matrix}[representation]
        X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
        init_centers = np.array([[0, 0], [1, 1]], dtype=dtype)
        # will be rescaled to [1.5, 0.5, 0.5, 1.5]
        if sw:
            sample_weight = [3, 1, 1, 3]
            expected_inertia = 0.375
            expected_centers = np.array([[0.125, 0], [0.875, 1]], dtype=dtype)
            expected_n_iter = 2
        else:
            sample_weight = None
            if norm == "L2":
                expected_inertia = 0.25
                expected_centers = np.array([[0.25, 0], [0.75, 1]], dtype=dtype)
                expected_n_iter = 2
            else:
                expected_inertia = 1.0
                expected_centers = np.array([[0.25, 0], [0.75, 1]], dtype=dtype)
                expected_n_iter = 1

        expected_labels = [0, 0, 1, 1]

        try:
            kmeans = KMeansL1L2(
                n_clusters=2, n_init=1, init=init_centers, algorithm=algo, norm=norm
            )
        except NotImplementedError as e:
            if "Only algorithm 'lloyd' is implemented" in str(e) and norm == "L1":
                return
            raise e

        try:
            kmeans.fit(X, sample_weight=sample_weight)
        except NotImplementedError as e:
            if "Non uniform weights are not implemented yet" in str(e) and norm == "L1":
                return
            if "Sparse matrix is not implemented" in str(e) and norm == "L1":
                return
            raise e

        assert_array_equal(kmeans.labels_, expected_labels)
        assert_almost_equal(kmeans.inertia_, expected_inertia)
        assert_array_almost_equal(kmeans.cluster_centers_, expected_centers)
        self.assertEqual(kmeans.n_iter_, expected_n_iter)

    @ignore_warnings(UserWarning)
    def test_kmeans_results(self):
        for representation, algo in [
            ("dense", "lloyd"),
            ("dense", "elkan"),
            ("sparse", "lloyd"),
        ]:
            for dtype in [np.float32, np.float64]:
                for norm in ["L1", "L2"]:
                    for sw in [False, True]:
                        with self.subTest(
                            c=representation, algo=algo, dtype=dtype, sw=sw, norm=norm
                        ):
                            self.do_test_kmeans_results(
                                representation, algo, dtype, norm, sw
                            )

    def _check_fitted_model(self, km):
        # check that the number of clusters centers and distinct labels match
        # the expectation
        centers = km.cluster_centers_
        self.assertEqual(
            centers.shape,
            (TestKMeansL1L2Sklearn.n_clusters, TestKMeansL1L2Sklearn.n_features),
        )

        labels = km.labels_
        self.assertEqual(np.unique(labels).shape[0], TestKMeansL1L2Sklearn.n_clusters)

        # check that the labels assignment are perfect (up to a permutation)
        self.assertEqual(
            v_measure_score(TestKMeansL1L2Sklearn.true_labels, labels), 1.0
        )
        self.assertGreater(km.inertia_, 0.0)

        # check error on dataset being too small
        assert_raise_message(
            ValueError,
            "n_samples=1 should be >= n_clusters=%d" % km.n_clusters,
            km.fit,
            [[0.0, 1.0]],
        )

    @ignore_warnings(UserWarning)
    def test_k_means_new_centers(self):
        # Explore the part of the code where a new center is reassigned
        X = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        labels = [0, 1, 2, 1, 1, 2]
        bad_centers = np.array([[+0, 1, 0, 0], [0.2, 0, 0.2, 0.2], [+0, 0, 0, 0]])

        km = KMeansL1L2(
            n_clusters=3, init=bad_centers, n_init=1, max_iter=10, random_state=1
        )
        for this_X in (X, sp.coo_matrix(X)):
            km.fit(this_X)
            this_labels = km.labels_
            # Reorder the labels so that the first instance is in cluster 0,
            # the second in cluster 1, ...
            this_labels = np.unique(this_labels, return_index=True)[1][this_labels]
            np.testing.assert_array_equal(this_labels, labels)

    @ignore_warnings(UserWarning)
    def test_k_means_plus_plus_init_not_precomputed(self):
        km = KMeansL1L2(
            init="k-means++",
            n_clusters=TestKMeansL1L2Sklearn.n_clusters,
            random_state=42,
        ).fit(TestKMeansL1L2Sklearn.X)
        self._check_fitted_model(km)

    @ignore_warnings(UserWarning)
    def test_k_means_random_init_not_precomputed(self):
        km = KMeansL1L2(
            init="random", n_clusters=TestKMeansL1L2Sklearn.n_clusters, random_state=42
        ).fit(TestKMeansL1L2Sklearn.X)
        self._check_fitted_model(km)


if __name__ == "__main__":
    unittest.main()
