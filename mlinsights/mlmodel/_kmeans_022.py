# pylint: disable=C0302
"""
@file
@brief Implements k-means with norms L1 and L2.
"""
import numpy
from scipy.sparse import issparse
# Source: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/cluster/_k_means_fast.pyx
from sklearn.utils.sparsefuncs_fast import assign_rows_csr  # pylint: disable=W0611,E0611
try:
    from sklearn.cluster._kmeans import _check_sample_weight
except ImportError:  # pragma: no cover
    from sklearn.cluster._kmeans import (
        _check_normalize_sample_weight as _check_sample_weight)
from sklearn.metrics.pairwise import pairwise_distances_argmin_min


def _labels_inertia_precompute_dense(norm, X, sample_weight, centers, distances):
    """
    Computes labels and inertia using a full distance matrix.

    This will overwrite the 'distances' array in-place.

    :param norm: 'l1' or 'l2'
    :param X:  numpy array, shape (n_sample, n_features)
        Input data.
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param centers: numpy array, shape (n_clusters, n_features)
        Cluster centers which data is assigned to.
    :param distances: numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.
    :return: labels : numpy array, dtype=numpy.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.
    :return: inertia : float
        Sum of squared distances of samples to their closest
        cluster center.
    """
    n_samples = X.shape[0]
    if norm == 'l2':
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=centers, metric='euclidean', metric_kwargs={'squared': True})
    elif norm == 'l1':
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=centers, metric='manhattan')
    else:  # pragma no cover
        raise NotImplementedError(
            "Not implemented for norm '{}'.".format(norm))
    # cython k-means code assumes int32 inputs
    labels = labels.astype(numpy.int32, copy=False)
    if n_samples == distances.shape[0]:
        # distances will be changed in-place
        distances[:] = mindist
    inertia = (mindist * sample_weight).sum()
    return labels, inertia


def _assign_labels_csr(X, sample_weight, x_squared_norms, centers,
                       labels, distances):
    """Compute label assignment and inertia for a CSR input
    Return the inertia (sum of squared distances to the centers).
    """
    if (distances is not None and
            distances.shape != (X.shape[0], )):
        raise ValueError(  # pragma: no cover
            "Dimension mismatch for distance got {}, expecting {}."
            "".format(distances.shape, (X.shape[0], centers.shape[0])))
    n_clusters = centers.shape[0]
    n_samples = X.shape[0]
    store_distances = 0
    inertia = 0.0

    if centers.dtype == numpy.float32:
        center_squared_norms = numpy.zeros(n_clusters, dtype=numpy.float32)
    else:
        center_squared_norms = numpy.zeros(n_clusters, dtype=numpy.float64)

    if n_samples == distances.shape[0]:
        store_distances = 1

    for center_idx in range(n_clusters):
        center_squared_norms[center_idx] = numpy.dot(
            centers[center_idx, :], centers[center_idx, :])

    for sample_idx in range(n_samples):
        min_dist = -1
        for center_idx in range(n_clusters):
            dist = 0.0
            # hardcoded: minimize euclidean distance to cluster center:
            # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
            dist += X[sample_idx, :] @ centers[center_idx, :].reshape((-1, 1))
            dist *= -2
            dist += center_squared_norms[center_idx]
            dist += x_squared_norms[sample_idx]
            dist *= sample_weight[sample_idx]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[sample_idx] = center_idx
                if store_distances:
                    distances[sample_idx] = dist
        inertia += min_dist

    return inertia


def _assign_labels_array(X, sample_weight, x_squared_norms, centers,
                         labels, distances):
    """Compute label assignment and inertia for a dense array
    Return the inertia (sum of squared distances to the centers).
    """
    n_clusters = centers.shape[0]
    n_samples = X.shape[0]
    store_distances = 0
    inertia = 0.0

    dtype = numpy.float32 if centers.dtype == numpy.float32 else numpy.float64
    center_squared_norms = numpy.zeros(n_clusters, dtype=dtype)

    if n_samples == distances.shape[0]:
        store_distances = 1

    for center_idx in range(n_clusters):
        center_squared_norms[center_idx] = numpy.dot(
            centers[center_idx, :], centers[center_idx, :])

    for sample_idx in range(n_samples):
        min_dist = -1
        for center_idx in range(n_clusters):
            dist = 0.0
            # hardcoded: minimize euclidean distance to cluster center:
            # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
            dist += numpy.dot(X[sample_idx, :], centers[center_idx, :])
            dist *= -2
            dist += center_squared_norms[center_idx]
            dist += x_squared_norms[sample_idx]
            dist *= sample_weight[sample_idx]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[sample_idx] = center_idx

        if store_distances:
            distances[sample_idx] = min_dist
        inertia += min_dist

    return inertia


def _labels_inertia_skl(X, sample_weight, x_squared_norms, centers,
                        precompute_distances=True, distances=None):
    """E step of the K-means EM algorithm.
    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.

    :param X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param x_squared_norms: array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.
    :param centers: float array, shape (k, n_features)
        The cluster centers.
    :param precompute_distances: boolean, default: True
        Precompute distances (faster but takes more memory).
    :param distances: float array, shape (n_samples,)
        Pre-allocated array to be filled in with each sample's distance
        to the closest center.
    :return: labels, int array of shape(n)
        The resulting assignment
    :return: inertia, float
        Sum of squared distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    sample_weight = _check_sample_weight(sample_weight, X)
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = numpy.full(n_samples, -1, numpy.int32)
    if distances is None:
        distances = numpy.zeros(shape=(0,), dtype=X.dtype)
    # distances will be changed in-place
    if issparse(X):
        inertia = _assign_labels_csr(
            X, sample_weight, x_squared_norms, centers, labels,
            distances=distances)
    else:
        if precompute_distances:
            return _labels_inertia_precompute_dense(
                norm='l2', X=X, sample_weight=sample_weight,
                centers=centers, distances=distances)
        inertia = _assign_labels_array(
            X, sample_weight, x_squared_norms, centers, labels,
            distances=distances)
    return labels, inertia


def _centers_dense(X, sample_weight, labels, n_clusters, distances):
    """
    M step of the K-means EM algorithm
    Computation of cluster centers / means.

    :param X: array-like, shape (n_samples, n_features)
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param labels: array of integers, shape (n_samples)
        Current label assignment
    :param n_clusters: int
        Number of desired clusters
    :param distances: array-like, shape (n_samples)
        Distance to closest cluster for each sample.
    :return: centers : array, shape (n_clusters, n_features)
        The resulting centers
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    dtype = X.dtype
    centers = numpy.zeros((n_clusters, n_features), dtype=dtype)
    weight_in_cluster = numpy.zeros((n_clusters,), dtype=dtype)

    for i in range(n_samples):
        c = labels[i]
        weight_in_cluster[c] += sample_weight[i]
    empty_clusters = numpy.where(weight_in_cluster == 0)[0]
    # maybe also relocate small clusters?

    if distances is not None and len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            far_index = far_from_centers[i]
            new_center = X[far_index] * sample_weight[far_index]
            centers[cluster_id] = new_center
            weight_in_cluster[cluster_id] = sample_weight[far_index]

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j] * sample_weight[i]

    centers /= weight_in_cluster[:, numpy.newaxis]

    return centers


def _centers_sparse(X, sample_weight, labels, n_clusters, distances):
    """
    M step of the K-means EM algorithm
    Computation of cluster centers / means.

    :param X: scipy.sparse.csr_matrix, shape (n_samples, n_features)
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param labels: array of integers, shape (n_samples)
        Current label assignment
    :param n_clusters: int
        Number of desired clusters
    :param distances: array-like, shape (n_samples)
        Distance to closest cluster for each sample.
    :return: centers, array, shape (n_clusters, n_features)
        The resulting centers
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    data = X.data
    indices = X.indices
    indptr = X.indptr

    dtype = X.dtype
    centers = numpy.zeros((n_clusters, n_features), dtype=dtype)
    weight_in_cluster = numpy.zeros((n_clusters,), dtype=dtype)
    for i in range(n_samples):
        c = labels[i]
        weight_in_cluster[c] += sample_weight[i]
    empty_clusters = numpy.where(weight_in_cluster == 0)[0]
    n_empty_clusters = empty_clusters.shape[0]

    # maybe also relocate small clusters?

    if n_empty_clusters > 0:
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1][:n_empty_clusters]
        assign_rows_csr(X, far_from_centers, empty_clusters, centers)

        for i in range(n_empty_clusters):
            weight_in_cluster[empty_clusters[i]] = 1

    for i in range(labels.shape[0]):
        curr_label = labels[i]
        for ind in range(indptr[i], indptr[i + 1]):
            j = indices[ind]
            centers[curr_label, j] += data[ind] * sample_weight[i]

    centers /= weight_in_cluster[:, numpy.newaxis]

    return centers
