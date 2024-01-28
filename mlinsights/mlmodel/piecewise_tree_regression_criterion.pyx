from libc.stdlib cimport calloc, free
cimport cython
cimport numpy as cnp

cnp.import_array()

from ._piecewise_tree_regression_common cimport (
    CommonRegressorCriterion,
    float64_t,
    intp_t,
)


cdef class SimpleRegressorCriterion(CommonRegressorCriterion):
    """
    Implements `mean square error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    criterion in a non efficient way. The code was inspired from
    `hellinger_distance_criterion.pyx
    <https://github.com/EvgeniDubov/hellinger-distance-criterion/blob/main/
    hellinger_distance_criterion.pyx>`_,
    `Cython example of exposing C-computed arrays in Python without data copies
    <http://gael-varoquaux.info/programming/
    cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html>`_,
    `_criterion.pyx
    <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx>`_.
    This implementation is not efficient but was made that way on purpose.
    It adds the features to the class.

    If the file does not compile or crashes, some explanations are given
    in :ref:`blog-internal-api-impurity-improvement`.
    """
    cdef float64_t* sample_w
    cdef float64_t* sample_wy
    cdef intp_t* sample_i
    cdef float64_t sample_sum_wy
    cdef float64_t sample_sum_w

    def __dealloc__(self):
        """Destructor."""
        free(self.sample_w)
        free(self.sample_wy)
        free(self.sample_i)
        self.sample_w = NULL
        self.sample_wy = NULL
        self.sample_i = NULL

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        self.n_outputs = n_outputs
        self.n_samples = n_samples

        # Allocate memory for the accumulators
        self.sample_w = NULL
        self.sample_wy = NULL
        self.sample_i = NULL

        # Criterion interface
        self.sample_weight = None
        self.sample_indices = None

        # allocation
        if self.sample_w == NULL:
            self.sample_w = <float64_t*> calloc(n_samples, sizeof(float64_t))
        if self.sample_wy == NULL:
            self.sample_wy = <float64_t*> calloc(n_samples, sizeof(float64_t))
        if self.sample_i == NULL:
            self.sample_i = <intp_t*> calloc(n_samples, sizeof(intp_t))

    def __str__(self):
        "usual"
        return (
            f"SimpleRegressorCriterion(n_outputs={self.n_outputs}, "
            f"n_samples={self.n_samples})"
        )

    @cython.boundscheck(False)
    cdef int init(self, const float64_t[:, ::1] y,
                  const float64_t[:] sample_weight,
                  float64_t weighted_n_samples,
                  const intp_t[:] sample_indices,
                  intp_t start, intp_t end) except -1 nogil:
        """
        This function is overwritten to check *y* and *X* size are the same.
        """
        if y.shape[0] != self.n_samples:
            return -1
        if y.shape[1] != 1:
            return -1
        return self.init_with_X(y, sample_weight, weighted_n_samples,
                                sample_indices, start, end)

    def printd(self):
        "debug print"
        rows = [str(self)]
        rows.append(f"  start={self.start}, pos={self.pos}, end={self.end}")
        rows.append(f"  weighted_n_samples={self.weighted_n_samples}")
        rows.append(f"  weighted_n_node_samples={self.weighted_n_node_samples}")
        rows.append(f"  sample_sum_w={self.sample_sum_w}")
        rows.append(f"  sample_sum_wy={self.sample_sum_wy}")
        rows.append(f"  weighted_n_left={self.weighted_n_left} "
                    f"weighted_n_right={self.weighted_n_right}")
        for ki in range(self.start, self.end):
            rows.append(f"  ki={ki}, sample_i={self.sample_i[ki]}, "
                        f"sample_w={self.sample_w[ki]}, sample_wy={self.sample_wy[ki]}")
        return "\n".join(rows)

    @cython.boundscheck(False)
    cdef int init_with_X(self,
                         const float64_t[:, ::1] y,
                         const float64_t[:] sample_weight,
                         float64_t weighted_n_samples,
                         const intp_t[:] sample_indices,
                         intp_t start, intp_t end) except -1 nogil:
        """
        Initializes the criterion.

        :param y: array-like, dtype=float64_t
            y is a buffer that can store values for n_outputs target variables
        :param sample_weight: array-like, dtype=float64_t
            The weight of each sample
        :param weighted_n_samples: float64_t
            The total weight of the samples being considered
        :param samples: array-like, dtype=float64_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        :param start: intp_t
            The first sample to be used on this node
        :param end: intp_t
            The last sample used on this node
        :return: 0 if everything is fine
        """
        cdef intp_t ki, ks
        self.start = start
        self.pos = start
        self.end = end
        self.weighted_n_samples = weighted_n_samples
        # Fatal Python error: __pyx_fatalerror: Acquisition count is 0
        self.y = y

        self.sample_sum_wy = 0.
        self.sample_sum_w = 0.

        if (
                (self.sample_w == NULL) or
                (self.sample_wy == NULL) or
                (self.sample_i == NULL)
        ):
            return -1

        # Filling accumulators.
        for ki in range(<int>start, <int>end):
            ks = sample_indices[ki]
            self.sample_i[ki] = ks
            self.sample_w[ki] = sample_weight[ks] if sample_weight is not None else 1.
            self.sample_wy[ki] = self.sample_w[ki] * y[ks, 0]
            self.sample_sum_wy += y[ks, 0] * self.sample_w[ki]
            self.sample_sum_w += self.sample_w[ki]

        self.weighted_n_node_samples = self.sample_sum_w
        return self.reset()

    cdef void _update_weights(self, intp_t start, intp_t end, intp_t old_pos,
                              intp_t new_pos) noexcept nogil:
        """
        Updates members `weighted_n_right` and `weighted_n_left`
        when `pos` changes.
        """
        self.weighted_n_right = 0
        self.weighted_n_left = 0
        for k in range(start, new_pos):
            self.weighted_n_left += self.sample_w[k]
        for k in range(new_pos, end):
            self.weighted_n_right += self.sample_w[k]

    cdef void _mean(self, intp_t start, intp_t end, float64_t *mean,
                    float64_t *weight) noexcept nogil:
        """
        Computes the mean of *y* between *start* and *end*.
        """
        if start == end:
            mean[0] = 0.
            return
        cdef float64_t m = 0.
        cdef float64_t w = 0.
        cdef int k
        for k in range(<int>start, <int>end):
            m += self.sample_wy[k]
            w += self.sample_w[k]
        weight[0] = w
        mean[0] = 0. if w == 0. else m / w

    @cython.boundscheck(False)
    cdef float64_t _mse(self, intp_t start, intp_t end, float64_t mean,
                        float64_t weight) noexcept nogil:
        """
        Computes mean square error between *start* and *end*
        assuming corresponding points are approximated by a constant.
        """
        if start == end:
            return 0.
        cdef float64_t squ = 0.
        cdef int k
        for k in range(<int>start, <int>end):
            squ += (self.y[self.sample_i[k], 0] - mean) ** 2 * self.sample_w[k]
        return 0. if weight == 0. else squ / weight
