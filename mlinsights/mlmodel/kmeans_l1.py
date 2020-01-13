# -*- coding: utf-8 -*-
"""
@file
@brief Implements k-means with norms L1 and L2.
"""
import numpy
from sklearn.cluster import KMeans


class KMeansL1L2(KMeans):
    """
    K-Means clustering with either norm L1 or L2.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'} or ndarray of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.

    precompute_distances : 'auto' or bool, default='auto'
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances.

        False : never precompute distances.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    norm : {"L1", "L2"}
        The norm *L2* is identical to :epkg:`KMeans`.
        Norm *L1* uses a complete different path.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto', norm='L2'):

        KMeans.__init__(self, n_clusters=n_clusters, init=init, n_init=n_init,
                        max_iter=max_iter, tol=tol,
                        precompute_distances=precompute_distances,
                        verbose=verbose, random_state=random_state,
                        copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)
        self.norm = norm.lower()

    def fit(self, X, y=None, sample_weight=None):
        """
        Computes k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        self
            Fitted estimator.
        """
        if self.norm == 'l2':
            KMeans.fit(self, X=X, y=y, sample_weight=sample_weight)
        elif self.norm == 'l1':
            self._fit_l1(X=X, y=y, sample_weight=sample_weight)
        else:
            raise NotImplemntedError(
                "Norm is not L1 or L2 but '{}'.".format(self.norm))
        return self

    def transform(self, X):
        """
        Transforms *X* to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        if self.norm == 'l2':
            return KMeans.transform(self, X)
        if self.norm == 'l1':
            return self._transform_l1(self, X)
        raise NotImplemntedError(
            "Norm is not L1 or L2 but '{}'.".format(self.norm))

    def _transform_L1(self, X):
        """
        Returns the distance of each point in *X* to
        every fit clusters.
        """
        check_is_fitted(self)
        X = self._check_test_data(X)
        stop

    def predict(self, X, sample_weight=None):
        """
        Predicts the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.norm == 'l2':
            return KMeans.predict(self, X)
        if self.norm == 'l1':
            return self._predict_l1(X, sample_weight=sample_weight)
        raise NotImplemntedError(
            "Norm is not L1 or L2 but '{}'.".format(self.norm))

    def _predict_L1(self, X, sample_weight=None):
        """
        Returns the distance of each point in *X* to
        every fit clusters.
        """
        n_samples = X.shape[0]

        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=centers, metric='minkowski', p=1)
        labels = labels.astype(np.int32, copy=False)
        return labels
