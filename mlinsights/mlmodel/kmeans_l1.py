# pylint: disable=C0302
"""
@file
@brief Implements k-means with norms L1 and L2.
"""
import warnings
import numpy
from scipy.sparse import issparse
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import _tolerance as _tolerance_skl
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import (
    euclidean_distances, manhattan_distances,
    pairwise_distances_argmin_min)
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import _num_samples, check_is_fitted
from sklearn.utils.extmath import stable_cumsum
try:
    from sklearn.cluster._kmeans import _check_sample_weight
except ImportError:  # pragma: no cover
    from sklearn.cluster._kmeans import (
        _check_normalize_sample_weight as _check_sample_weight)
from ._kmeans_022 import (
    _labels_inertia_skl,
    _labels_inertia_precompute_dense)


def _k_init(norm, X, n_clusters, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    :param norm: `l1` or `l2`
        manhattan or euclidean distance
    :param X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=numpy.float64).
    :param n_clusters: integer
        The number of seeds to choose
    :param random_state: int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.
    :param n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    """
    n_samples, n_features = X.shape

    centers = numpy.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(numpy.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    if norm.lower() == 'l2':
        dist_fct = lambda x, y: euclidean_distances(x, y, squared=True)
    elif norm.lower() == 'l1':
        dist_fct = lambda x, y: manhattan_distances(x, y)
    else:
        raise NotImplementedError(  # pragma no cover
            "norm must be 'l1' or 'l2' not '{}'.".format(norm))

    closest_dist_sq = dist_fct(centers[0, numpy.newaxis], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = numpy.searchsorted(stable_cumsum(closest_dist_sq),
                                           rand_vals)
        numpy.clip(candidate_ids, None, closest_dist_sq.size - 1,
                   out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = dist_fct(X[candidate_ids], X)

        # update closest distances squared and potential for each candidate
        numpy.minimum(closest_dist_sq, distance_to_candidates,
                      out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = numpy.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers


def _init_centroids(norm, X, k, init, random_state=None,
                    init_size=None):
    """Compute the initial centroids

    :param norm: 'l1' or 'l2'
    :param X: array, shape (n_samples, n_features)
    :param k: int
        number of centroids
    :param init: {'k-means++', 'random' or ndarray or callable} optional
        Method for initialization
    :param random_state: int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    :param init_size: int, optional
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.
    :return: centers, array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if init_size is not None and init_size < n_samples:
        if init_size < k:  # pragma: no cover
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
        raise ValueError(  # pragma: no cover
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if isinstance(init, str) and init == 'k-means++':
        centers = _k_init(norm, X, k, random_state=random_state)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        # ensure that the centers have the same dtype as X
        # this is a requirement of fused types of cython
        centers = numpy.array(init, dtype=X.dtype)
    elif callable(init):
        centers = init(norm, X, k, random_state=random_state)
        centers = numpy.asarray(centers, dtype=X.dtype)
    else:
        raise ValueError(  # pragma: no cover
            "init parameter for the k-means should "
            "be 'k-means++' or 'random' or an ndarray, "
            "'%s' (type '%s') was passed." % (init, type(init)))

    if issparse(centers):
        centers = centers.toarray()

    def _validate_center_shape(X, k, centers):
        """Check if centers is compatible with X and n_clusters"""
        if centers.shape[0] != k:
            raise ValueError(  # pragma: no cover
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {k}.")
        if centers.shape[1] != X.shape[1]:
            raise ValueError(  # pragma: no cover
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}.")

    _validate_center_shape(X, k, centers)
    return centers


def _centers_dense(X, sample_weight, labels, n_clusters, distances,
                   X_sort_index):
    """
    M step of the K-means EM algorithm.
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
    :param X_sort_index: array-like, shape (n_samples, n_features)
        index of each feature in all features
    :return: centers, array, shape (n_clusters, n_features)
        The resulting centers
    """
    dtype = X.dtype
    n_features = X.shape[1]
    n_samples = X.shape[0]

    centers = numpy.zeros((n_clusters, n_features), dtype=dtype)
    weight_in_cluster = numpy.zeros((n_clusters,), dtype=dtype)

    for i in range(n_samples):
        c = labels[i]
        weight_in_cluster[c] += sample_weight[i]
    empty_clusters = numpy.where(weight_in_cluster == 0)[0]

    if len(empty_clusters) > 0:  # pragma: no cover
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            far_index = far_from_centers[i]
            new_center = X[far_index] * sample_weight[far_index]
            centers[cluster_id] = new_center
            weight_in_cluster[cluster_id] = sample_weight[far_index]

    if sample_weight.min() == sample_weight.max():
        # to optimize
        for i in range(n_clusters):
            sub = X[labels == i]
            med = numpy.median(sub, axis=0)
            centers[i, :] = med
    else:
        raise NotImplementedError(  # pragma: no cover
            "Non uniform weights are not implemented yet as "
            "the cost would be very high. "
            "See https://en.wikipedia.org/wiki/Weighted_median#Algorithm.")
    return centers


def _kmeans_single_lloyd(norm, X, sample_weight, n_clusters, max_iter=300,
                         init='k-means++', verbose=False,
                         random_state=None, tol=1e-4,
                         precompute_distances=True):
    """
    A single run of k-means, assumes preparation completed prior.

    :param norm: 'l1' or 'l2'
    :param X: array-like of floats, shape (n_samples, n_features)
        The observations to cluster.
    :param n_clusters: int
        The number of clusters to form as well as the number of
        centroids to generate.
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param max_iter: int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.
    :param init: {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    :param tol: float, optional
        The relative increment in the results before declaring convergence.
    :param verbose: boolean, optional
        Verbosity mode
    :param precompute_distances: boolean, default: True
        Precompute distances (faster but takes more memory).
    :param random_state: int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    :return: centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.
    :return: label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.
    :return: inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).
    :return: n_iter : int
        Number of iterations run.
    """
    random_state = check_random_state(random_state)

    sample_weight = _check_sample_weight(sample_weight, X)

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(
        norm, X, n_clusters, init, random_state=random_state)
    if verbose:  # pragma no cover
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = numpy.zeros(shape=(X.shape[0],), dtype=X.dtype)
    X_sort_index = numpy.argsort(X, axis=0)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(norm, X, sample_weight, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)

        # computation of the means is also called the M-step of EM
        centers = _centers_dense(X, sample_weight, labels, n_clusters, distances,
                                 X_sort_index)

        if verbose:  # pragma no cover
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = numpy.sum(
            numpy.abs(centers_old - centers).ravel())
        if center_shift_total <= tol:
            if verbose:  # pragma no cover
                print("Converged at iteration %d: "
                      "center shift %r within tolerance %r"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(norm, X, sample_weight, best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances)

    return best_labels, best_inertia, best_centers, i + 1


def _labels_inertia(norm, X, sample_weight, centers,
                    precompute_distances=True, distances=None):
    """
    E step of the K-means EM algorithm.

    Computes the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.

    :param norm: 'l1' or 'l2'
    :param X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param centers: float array, shape (k, n_features)
        The cluster centers.
    :param precompute_distances: boolean, default: True
        Precompute distances (faster but takes more memory).
    :param distances: existing distances
    :return: labels : int array of shape(n)
        The resulting assignment
    :return: inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    if norm == 'l2':
        return _labels_inertia_skl(
            X, sample_weight=sample_weight, centers=centers,
            precompute_distances=precompute_distances,
            x_squared_norms=None)

    sample_weight = _check_sample_weight(sample_weight, X)
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    if distances is None:
        distances = numpy.zeros(shape=(0,), dtype=X.dtype)
    # distances will be changed in-place
    if issparse(X):
        raise NotImplementedError(  # pragma no cover
            "Sparse matrix is not implemented for norm 'l1'.")
    if precompute_distances:
        return _labels_inertia_precompute_dense(
            norm=norm, X=X, sample_weight=sample_weight,
            centers=centers, distances=distances)
    raise NotImplementedError(  # pragma no cover
        "precompute_distances is False, not implemented for norm 'l1'.")


def _tolerance(norm, X, tol):
    """Return a tolerance which is independent of the dataset"""
    if norm == 'l2':
        return _tolerance_skl(X, tol)
    if norm == 'l1':
        variances = numpy.sum(numpy.abs(X), axis=0) / X.shape[0]
        return variances.sum()
    raise NotImplementedError(  # pragma no cover
        "not implemented for norm '{}'.".format(norm))


class KMeansL1L2(KMeans):
    """
    K-Means clustering with either norm L1 or L2.
    See notebook :ref:`kmeansl1rst` for an example.

    :param n_clusters: int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    :param init: {'k-means++', 'random'} or ndarray of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    :param n_init: int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    :param max_iter: int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    :param tol: float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    :param precompute_distances: 'auto' or bool, default='auto'
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances.

        False : never precompute distances.

    :param verbose: int, default=0
        Verbosity mode.
    :param random_state: int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    :param copy_x: bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.
    :param n_jobs: int, default=None
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    :param algorithm: {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.
    :param norm: {"L1", "L2"}
        The norm *L2* is identical to :epkg:`KMeans`.
        Norm *L1* uses a complete different path.

    Fitted attributes:

    * `cluster_centers_`: ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    * `labels_`: ndarray of shape (n_samples,)
        Labels of each point
    * `inertia_`: float
        Sum of squared distances of samples to their closest cluster center.
    * `n_iter_`: int
        Number of iterations run.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='full', norm='L2'):

        KMeans.__init__(self, n_clusters=n_clusters, init=init, n_init=n_init,
                        max_iter=max_iter, tol=tol,
                        precompute_distances=precompute_distances,
                        verbose=verbose, random_state=random_state,
                        copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)
        self.norm = norm.lower()
        if self.norm == 'l1' and self.algorithm != 'full':
            raise NotImplementedError(  # pragma no cover
                "Only algorithm 'full' is implemented with norm 'l1'.")

    def fit(self, X, y=None, sample_weight=None):
        """
        Computes k-means clustering.

        :param X: array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        :param y: Ignored
            Not used, present here for API consistency by convention.
        :param sample_weight: array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        :return: self
            Fitted estimator.
        """
        if self.norm == 'l2':
            KMeans.fit(self, X=X, y=y, sample_weight=sample_weight)
        elif self.norm == 'l1':
            self._fit_l1(X=X, y=y, sample_weight=sample_weight)
        else:
            raise NotImplementedError(  # pragma no cover
                "Norm is not L1 or L2 but '{}'.".format(self.norm))
        return self

    def _fit_l1(self, X, y=None, sample_weight=None):
        """
        Computes k-means clustering with norm `'l1'`.

        :param X: array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        :param y: Ignored
            Not used, present here for API consistency by convention.
        :param sample_weight: array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        :return: self
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        n_init = self.n_init
        if n_init <= 0:
            raise ValueError(  # pragma no cover
                "Invalid number of initializations."
                " n_init=%d must be bigger than zero." % n_init)

        if self.max_iter <= 0:
            raise ValueError(  # pragma no cover
                'Number of iterations should be a positive number,'
                ' got %d instead' % self.max_iter)

        # avoid forcing order when copy_x=False
        order = "C" if self.copy_x else None
        X = check_array(X, accept_sparse='csr', dtype=[numpy.float64, numpy.float32],
                        order=order, copy=self.copy_x)
        # verify that the number of samples given is larger than k
        if _num_samples(X) < self.n_clusters:
            raise ValueError(  # pragma no cover
                "n_samples=%d should be >= n_clusters=%d" % (
                    _num_samples(X), self.n_clusters))

        tol = _tolerance(self.norm, X, self.tol)

        # If the distances are precomputed every job will create a matrix of
        # shape (n_clusters, n_samples). To stop KMeans from eating up memory
        # we only activate this if the created matrix is guaranteed to be
        # under 100MB. 12 million entries consume a little under 100MB if they
        # are of type double.
        precompute_distances = self.precompute_distances
        if precompute_distances == 'auto':
            n_samples = X.shape[0]
            precompute_distances = (self.n_clusters * n_samples) < 12e6
        elif isinstance(precompute_distances, bool):  # pragma: no cover
            pass
        else:
            raise ValueError(  # pragma no cover
                "precompute_distances should be 'auto' or True/False"
                ", but a value of %r was passed" % precompute_distances)

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype.type, copy=True)
            if hasattr(self, '_validate_center_shape'):
                self._validate_center_shape(  # pylint: disable=E1101
                    X, init)

            if n_init != 1:
                warnings.warn(  # pragma: no cover
                    'Explicit initial center position passed: '
                    'performing only one init in k-means instead of n_init=%d'
                    % n_init, RuntimeWarning, stacklevel=2)
                n_init = 1

        best_labels, best_inertia, best_centers = None, None, None
        algorithm = self.algorithm
        if self.n_clusters == 1:
            # elkan doesn't make sense for a single cluster, full will produce
            # the right result.
            algorithm = "full"  # pragma: no cover
        if algorithm == "auto":
            algorithm = "full"  # pragma: no cover
        if algorithm == "full":
            kmeans_single = _kmeans_single_lloyd
        else:
            raise ValueError(  # pragma no cover
                "Algorithm must be 'auto', 'full' or 'elkan', got"
                " %s" % str(algorithm))

        seeds = random_state.randint(numpy.iinfo(numpy.int32).max, size=n_init)
        if effective_n_jobs(self.n_jobs) == 1:
            # For a single thread, less memory is needed if we just store one
            # set of the best results (as opposed to one set per run per
            # thread).
            for seed in seeds:
                # run a k-means once
                labels, inertia, centers, n_iter_ = kmeans_single(
                    self.norm, X, sample_weight, self.n_clusters,
                    max_iter=self.max_iter, init=init, verbose=self.verbose,
                    precompute_distances=precompute_distances, tol=tol,
                    random_state=seed)
                # determine if these results are the best so far
                if best_inertia is None or inertia < best_inertia:
                    best_labels = labels.copy()
                    best_centers = centers.copy()
                    best_inertia = inertia
                    best_n_iter = n_iter_
        else:
            # parallelisation of k-means runs
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(kmeans_single)(
                    self.norm, X, sample_weight, self.n_clusters,
                    max_iter=self.max_iter, init=init,
                    verbose=self.verbose, tol=tol,
                    precompute_distances=precompute_distances,
                    # Change seed to ensure variety
                    random_state=seed
                )
                for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = numpy.argmin(inertia)
            best_labels = labels[best]
            best_inertia = inertia[best]
            best_centers = centers[best]
            best_n_iter = n_iters[best]

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(  # pragma no cover
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2)

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def transform(self, X):
        """
        Transforms *X* to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        :return: X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        if self.norm == 'l2':
            return KMeans.transform(self, X)
        if self.norm == 'l1':
            return self._transform_l1(X)
        raise NotImplementedError(  # pragma no cover
            "Norm is not L1 or L2 but '{}'.".format(self.norm))

    def _transform_l1(self, X):
        """
        Returns the distance of each point in *X* to
        every fit clusters.
        """
        check_is_fitted(self)
        X = self._check_test_data(X)
        return manhattan_distances(X, self.cluster_centers_)

    def predict(self, X, sample_weight=None):
        """
        Predicts the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        :param sample_weight: array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None), unused here
        :return: labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.norm == 'l2':
            return KMeans.predict(self, X)
        if self.norm == 'l1':
            return self._predict_l1(X, sample_weight=sample_weight)
        raise NotImplementedError(  # pragma no cover
            "Norm is not L1 or L2 but '{}'.".format(self.norm))

    def _predict_l1(self, X, sample_weight=None, return_distances=False):
        """
        Returns the distance of each point in *X* to
        every fit clusters.

        :param X: features
        :param sample_weight: (unused)
        :param return_distances: returns distances as well
        :return: labels or `labels, distances`
        """
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=self.cluster_centers_, metric='manhattan')
        labels = labels.astype(numpy.int32, copy=False)
        if return_distances:
            return labels, mindist
        return labels
