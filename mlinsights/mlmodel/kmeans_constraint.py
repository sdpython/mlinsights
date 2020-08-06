# -*- coding: utf-8 -*-
"""
@file
@brief Implémente la classe @see cl ConstraintKMeans.
"""
import numpy
from scipy.spatial import Delaunay  # pylint: disable=E0611
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from ._kmeans_constraint_ import constraint_kmeans, constraint_predictions


class ConstraintKMeans(KMeans):
    """
    Defines a constraint :epkg:`k-means`.
    Clusters are modified to have an equal size.
    The algorithm is initialized with a regular :epkg:`KMeans`
    and continues with a modified version of it.

    Computing the predictions offer a choice.
    The first one is to keep the predictions
    from the regular :epkg:`k-means`
    algorithm but with the balanced clusters.
    The second is to compute balanced predictions
    over the test set. That implies that the predictions
    for the same observations might change depending
    on the set it belongs to.

    The parameter *strategy* determines how
    obseervations should be assigned to a cluster.
    The value can be:

    * ``'distance'``: observations are ranked by distance to a cluster,
      the algorithm assigns first point to the closest center unless it reached
      the maximum size, it deals first with the further point and maps it to
      the closest center
    * ``'gain'``: follows the algorithm described at
       see `Same-size k-Means Variation
       <https://elki-project.github.io/tutorial/same-size_k_means>`_,
    * ``'weights'``: estimates weights attached to each cluster,
        it weights the distance to each cluster in order
        to balance the number of points mapped to every cluster,
        the strategy uses a learning rate.

    The first two strategies cannot reach a good compromise
    without using function @see fn _switch_clusters which
    tries every switch between clusters: two points
    change clusters. It keeps the number of points and checks
    that the inertia is reduced.
    """

    _strategy_value = {'distance', 'gain', 'weights'}

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=500,
                 tol=0.0001, precompute_distances='deprecated', verbose=0,
                 random_state=None, copy_x=True, n_jobs=1, algorithm='auto',
                 balanced_predictions=False, strategy='gain', kmeans0=True,
                 learning_rate=1., history=False):
        """
        @param      n_clusters              number of clusters
        @param      init                    used by :epkg:`k-means`
        @param      n_init                  used by :epkg:`k-means`
        @param      max_iter                used by :epkg:`k-means`
        @param      tol                     used by :epkg:`k-means`
        @param      precompute_distances    used by :epkg:`k-means`
        @param      verbose                 used by :epkg:`k-means`
        @param      random_state            used by :epkg:`k-means`
        @param      copy_x                  used by :epkg:`k-means`
        @param      n_jobs                  used by :epkg:`k-means`
        @param      algorithm               used by :epkg:`k-means`
        @param      balanced_predictions    produced balanced prediction
                                            or the regular ones
        @param      strategy                strategy or algorithm used to abide
                                            by the constraint
        @param      kmeans0                 if True, applies *k-means* algorithm first
        @param      history                 keeps centers accress iterations
        @param      learning_rate           learning rate, used by strategy `'weights'`
        """
        KMeans.__init__(self, n_clusters=n_clusters, init=init, n_init=n_init,
                        max_iter=max_iter, tol=tol, precompute_distances=precompute_distances,
                        verbose=verbose, random_state=random_state, copy_x=copy_x,
                        n_jobs=n_jobs, algorithm=algorithm)
        self.balanced_predictions = balanced_predictions
        self.strategy = strategy
        self.kmeans0 = kmeans0
        self.history = history
        self._n_threads = None
        self.learning_rate = learning_rate
        if strategy not in ConstraintKMeans._strategy_value:
            raise ValueError('strategy must be in {0}'.format(
                ConstraintKMeans._strategy_value))
        if precompute_distances == 'deprecated':
            km = KMeans()
            if km.precompute_distances != precompute_distances:
                self.precompute_distances = km.precompute_distances

    def fit(self, X, y=None, sample_weight=None, fLOG=None):
        """
        Compute k-means clustering.

        :param X: array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        :param y: Ignored
        :param sample_weight: sample weight
        :param fLOG: logging function
        """
        max_iter = self.max_iter
        self.max_iter //= 2
        if self.kmeans0:
            KMeans.fit(self, X, y, sample_weight=sample_weight)
            state = None
        else:
            state = numpy.random.RandomState(  # pylint: disable=E1101
                self.random_state)
            labels = state.randint(
                0, self.n_clusters, X.shape[0], dtype=numpy.int32)
            centers = numpy.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
            choice = state.randint(0, self.n_clusters, self.n_clusters)
            for i, c in enumerate(choice):
                centers[i, :] = X[c, :]
            self.labels_ = labels
            self.cluster_centers_ = centers
            self.inertia_ = float(X.shape[0])
            self.n_iter_ = 0

        self.max_iter = max_iter
        return self.constraint_kmeans(
            X, sample_weight=sample_weight, state=state,
            learning_rate=self.learning_rate,
            history=self.history, fLOG=fLOG)

    def constraint_kmeans(self, X, sample_weight=None, state=None,
                          learning_rate=1., history=False, fLOG=None):
        """
        Completes the constraint k-means.

        @param      X               features
        @param      sample_weight   sample weight
        @param      state           state
        @param      history         keeps evolution of centers
        @param      fLOG            logging function
        """
        labels, centers, inertia, weights, iter_, all_centers = constraint_kmeans(
            X, self.labels_, sample_weight, self.cluster_centers_,
            inertia=self.inertia_, iter=self.n_iter_,
            max_iter=self.max_iter, verbose=self.verbose,
            strategy=self.strategy, state=state,
            learning_rate=learning_rate, history=history,
            fLOG=fLOG)
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.cluster_centers_iter_ = (
            None if len(all_centers) == 0 else numpy.dstack(all_centers))
        self.inertia_ = inertia
        self.n_iter_ = iter_
        self.weights_ = weights
        return self

    def predict(self, X, sample_weight=None):
        """
        Computes the predictions.

        @param      X       features.
        @return             prediction
        """
        if self.weights_ is None:
            if self.balanced_predictions:
                labels, _, __ = constraint_predictions(
                    X, self.cluster_centers_, strategy=self.strategy + '_p')
                return labels
            return KMeans.predict(self, X, sample_weight=sample_weight)
        else:
            if self.balanced_predictions:
                raise RuntimeError(  # pragma: no cover
                    "balanced_predictions and weights_ cannot be used together.")
            return KMeans.predict(self, X, sample_weight=sample_weight)

    def transform(self, X):
        """
        Computes the predictions.

        @param      X       features.
        @return             prediction
        """
        if self.weights_ is None:
            if self.balanced_predictions:
                labels, distances, __ = constraint_predictions(
                    X, self.cluster_centers_, strategy=self.strategy)
                # We remove small distances than the chosen clusters
                # due to the constraint, we choose max*2 instead.
                mx = distances.max() * 2
                for i, l in enumerate(labels):
                    mi = distances[i, l]
                    mmi = distances[i, :].min()
                    if mi > mmi:
                        # numpy.nan would be best
                        distances[i, distances[i, :] < mi] = mx
                return distances
            return KMeans.transform(self, X)
        else:
            if self.balanced_predictions:
                raise RuntimeError(  # pragma: no cover
                    "balanced_predictions and weights_ cannot be used together.")
            res = KMeans.transform(self, X)
            res *= self.weights_.reshape((1, -1))
            return res

    def score(self, X, y=None, sample_weight=None):
        """
        Returns the distances to all clusters.

        @param      X               features
        @param      y               unused
        @param      sample_weight   sample weight
        @return                     distances
        """
        if self.weights_ is None:
            if self.balanced_predictions:
                _, __, dist_close = constraint_predictions(
                    X, self.cluster_centers_, strategy=self.strategy)
                return dist_close
            res = euclidean_distances(self.cluster_centers_, X, squared=True)
        else:
            if self.balanced_predictions:
                raise RuntimeError(  # pragma: no cover
                    "balanced_predictions and weights_ cannot be used together.")
            res = euclidean_distances(X, self.cluster_centers_, squared=True)
            res *= self.weights_.reshape((1, -1))
        return res.max(axis=1)

    def cluster_edges(self):
        """
        Computes edges between clusters based on a
        `Delaunay <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.spatial.Delaunay.html>`_
        graph.
        """
        tri = Delaunay(self.cluster_centers_)
        triangles = tri.simplices  # pylint: disable=E1101
        edges = set()
        for row in triangles:
            for j in range(1, row.shape[-1]):
                a, b = row[j - 1:j + 1]
                if a < b:
                    edges.add((a, b))
                else:
                    edges.add((b, a))
            a, b = row[0], row[-1]
            if a < b:
                edges.add((a, b))
            else:
                edges.add((b, a))
        return edges
