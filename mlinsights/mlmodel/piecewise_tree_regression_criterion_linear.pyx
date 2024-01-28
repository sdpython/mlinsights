cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

cimport scipy.linalg.cython_lapack as cython_lapack
from ._piecewise_tree_regression_common cimport (
    CommonRegressorCriterion,
    intp_t,
    float64_t,
)


cdef class LinearRegressorCriterion(CommonRegressorCriterion):
    """
    Criterion which computes the mean square error
    assuming points falling into one node are approximated
    by a line (linear regression).
    The implementation follows the same
    design used in :class:`SimpleRegressorCriterion
    <mlinsights.mlmodel.piecewise_tree_regression_criterion.SimpleRegressorCriterion>`
    and is even slow as the criterion is more complex to compute.

    If the file does not compile or crashes, some explanations are given
    in :ref:`blog-internal-api-impurity-improvement`.
    """
    cdef intp_t n_features
    cdef const float64_t[:, ::1] sample_X
    cdef float64_t* sample_w
    cdef float64_t* sample_y
    cdef float64_t* sample_wy
    cdef float64_t* sample_f
    cdef float64_t* sample_pC
    cdef float64_t* sample_pS
    cdef float64_t* sample_work
    cdef intp_t* sample_i
    cdef float64_t* sample_f_buffer

    cdef float64_t sample_sum_wy
    cdef float64_t sample_sum_w
    cdef intp_t nbvar
    cdef intp_t nbrows
    cdef intp_t work

    def __dealloc__(self):
        """Destructor."""
        free(self.sample_w)
        free(self.sample_y)
        free(self.sample_wy)
        free(self.sample_i)
        free(self.sample_f)
        free(self.sample_f_buffer)
        free(self.sample_pC)
        free(self.sample_pS)
        free(self.sample_work)
        self.sample_w = NULL
        self.sample_y = NULL
        self.sample_wy = NULL
        self.sample_i = NULL
        self.sample_f = NULL
        self.sample_f_buffer = NULL
        self.sample_pC = NULL
        self.sample_pS = NULL
        self.sample_work = NULL

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __cinit__(self, intp_t n_outputs, const float64_t[:, ::1] X):
        self.n_outputs = n_outputs
        self.sample_X = X
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # Allocate memory for the accumulators
        self.sample_w = NULL
        self.sample_y = NULL
        self.sample_wy = NULL
        self.sample_i = NULL
        self.sample_f = NULL
        self.sample_f_buffer = NULL
        self.sample_pC = NULL
        self.sample_pS = NULL
        self.sample_work = NULL

        # Criterion interface
        self.sample_weight = None
        self.sample_indices = None

        # allocation
        if self.sample_w == NULL:
            self.sample_w = <float64_t*> calloc(self.n_samples, sizeof(float64_t))
        if self.sample_wy == NULL:
            self.sample_wy = <float64_t*> calloc(self.n_samples, sizeof(float64_t))
        if self.sample_y == NULL:
            self.sample_y = <float64_t*> calloc(self.n_samples, sizeof(float64_t))
        if self.sample_i == NULL:
            self.sample_i = <intp_t*> calloc(self.n_samples, sizeof(intp_t))
        if self.sample_f == NULL:
            self.sample_f = <float64_t*> calloc(
                self.n_samples * (self.n_features + 1), sizeof(float64_t)
            )

        self.nbvar = self.n_features + 1
        self.nbrows = self.n_samples
        self.work = <intp_t>(min(self.nbrows, self.nbvar) * <intp_t>3 +
                             max(max(self.nbrows, self.nbvar),
                                 min(self.nbrows, self.nbvar) * <intp_t>2))
        if self.sample_f_buffer == NULL:
            self.sample_f_buffer = <float64_t*> calloc(
                self.n_samples * self.nbvar, sizeof(float64_t)
            )
        if self.sample_pC == NULL:
            self.sample_pC = <float64_t*> calloc(
                max(self.nbrows, self.nbvar), sizeof(float64_t)
            )
        if self.sample_work == NULL:
            self.sample_work = <float64_t*> calloc(self.work, sizeof(float64_t))
        if self.sample_pS == NULL:
            self.sample_pS = <float64_t*> calloc(self.nbvar, sizeof(float64_t))

    def __deepcopy__(self, memo=None):
        """
        This does not a copy but mostly creates a new instance
        of the same criterion initialized with the same data.
        """
        inst = self.__class__(self.n_outputs, self.sample_X)
        return inst

    @staticmethod
    def create(const float64_t[:, ::1] X, const float64_t[:, ::1] y,
               const float64_t[::1] sample_weight=None):
        """
        Initializes the criterion.

        :param X: features
        :param y: target
        :param sample_weight: sample weight
        :return: an instance of :class:`LinearRegressorCriterion`
        """
        cdef intp_t i
        cdef const float64_t[:] ws
        cdef float64_t sum
        cdef intp_t[:] parr = np.empty(y.shape[0], dtype=np.int64)
        for i in range(0, y.shape[0]):
            parr[i] = i
        if sample_weight is None:
            sum = <float64_t>y.shape[0]
            ws = None
        else:
            sum = sample_weight.sum()
            ws = sample_weight

        obj = LinearRegressorCriterion(1 if len(y.shape) <= 1 else y.shape[0], X)
        obj.init(y, ws, sum, parr, 0, y.shape[0])
        return obj

    cdef int init(self, const float64_t[:, ::1] y,
                  const float64_t[:] sample_weight,
                  float64_t weighted_n_samples,
                  const intp_t[:] sample_indices,
                  intp_t start, intp_t end) except -1 nogil:
        """
        This function is overwritten to check *y* and *X* size are the same.
        This API changed in 0.21.
        It changed again in scikit-learn 1.2 to replace `float64_t*` into `DOUBLE[:]`.
        """
        if y.shape[0] != self.n_samples:
            raise ValueError(
                "n_samples={} -- y.shape={}".format(self.n_samples, y.shape)
            )
        if y.shape[0] != self.sample_X.shape[0]:
            raise ValueError(
                "X.shape={} -- y.shape={}".format(self.sample_X.shape, y.shape)
            )
        if y.shape[1] != 1:
            raise ValueError("This class only works for a single vector.")
        return self.init_with_X(self.sample_X, y, sample_weight, weighted_n_samples,
                                sample_indices, start, end)

    @cython.boundscheck(False)
    cdef int init_with_X(self, const float64_t[:, ::1] X,
                         const float64_t[:, ::1] y,
                         const float64_t[:] sample_weight,
                         float64_t weighted_n_samples,
                         const intp_t[:] sample_indices,
                         intp_t start, intp_t end) except -1 nogil:
        """
        Initializes the criterion.

        :param X: array-like, features, dtype=float64_t
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
        """
        cdef intp_t ki, ks, idx, c

        self.start = start
        self.pos = start
        self.end = end
        self.weighted_n_samples = weighted_n_samples
        self.y = y

        self.sample_sum_wy = 0.
        self.sample_sum_w = 0.

        # Filling accumulators.
        idx = start * self.nbvar
        for ki in range(<int>start, <int>end):
            ks = sample_indices[ki]
            self.sample_i[ki] = ks
            self.sample_w[ki] = sample_weight[ks] if sample_weight is not None else 1.
            self.sample_wy[ki] = self.sample_w[ki] * y[ks, 0]
            self.sample_y[ki] = y[ks, 0]
            self.sample_sum_wy += y[ks, 0] * self.sample_w[ki]
            self.sample_sum_w += self.sample_w[ki]
            for c in range(0, <int>self.nbvar-1):
                self.sample_f[idx] = X[ks, c]
                idx += 1
            self.sample_f[idx] = 1.
            idx += 1

        self.weighted_n_node_samples = self.sample_sum_w
        self.reset()
        if self.weighted_n_node_samples == 0:
            raise ValueError(
                f"self.weighted_n_node_samples is null, "
                f"first weight is {self.sample_w[0]}."
            )
        return 0

    cdef void _mean(self, intp_t start, intp_t end, float64_t *mean,
                    float64_t *weight) noexcept nogil:
        """
        Computes mean between *start* and *end*.
        """
        if start == end:
            mean[0] = 0.
            return
        cdef float64_t m = 0.
        cdef float64_t w = 0.
        cdef intp_t k
        for k in range(<int>start, <int>end):
            m += self.sample_wy[k]
            w += self.sample_w[k]
        weight[0] = w
        mean[0] = 0. if w == 0. else m / w

    cdef void _reglin(self, intp_t start, intp_t end, int low_rank) noexcept nogil:
        """
        Solves the linear regression between *start* and *end*
        assuming corresponding points are approximated by a line.
        *mean* is unused but could be if the inverse does not exist.
        The solution is the vector ``self.sample_pC[:self.nbvar]``.
        """
        cdef intp_t i, j, idx, pos
        cdef float64_t w
        cdef float64_t* sample_f_buffer = self.sample_f_buffer
        pos = 0
        for j in range(self.nbvar):
            idx = start * self.nbvar + j
            for i in range(<int>start, <int>end):
                w = self.sample_w[i]
                sample_f_buffer[pos] = self.sample_f[idx] * w
                idx += self.nbvar
                pos += 1

        cdef float64_t* pC = self.sample_pC
        for i in range(<int>start, <int>end):
            pC[i-start] = self.sample_wy[i]

        cdef int col = <int>self.nbvar
        cdef int row = <int>(end - start)
        cdef int info
        cdef int nrhs = 1
        cdef int lda = row
        cdef int ldb = row
        cdef float64_t rcond = -1
        cdef int rank
        cdef int work = <int>self.work

        if row < col:
            if low_rank:
                ldb = col
            else:
                return
        cython_lapack.dgelss(&row, &col, &nrhs,                 # 1-3
                             sample_f_buffer, &lda, pC, &ldb,   # 4-7
                             self.sample_pS, &rcond, &rank,     # 8-10
                             self.sample_work, &work, &info)    # 11-13

    cdef float64_t _mse(self, intp_t start, intp_t end, float64_t mean,
                        float64_t weight) noexcept nogil:
        """
        Computes mean square error between *start* and *end*
        assuming corresponding points are approximated by a line.
        *mean* is unused but could be if the inverse does not exist.
        """
        if end - start <= self.nbvar:
            # More coefficients than the number of observations.
            return 0.

        self._reglin(start, end, 0)

        cdef float64_t* pC = self.sample_pC
        cdef intp_t j, idx

        # replaces what follows by gemm
        cdef float64_t squ = 0.
        cdef float64_t d
        cdef intp_t k
        idx = start * self.nbvar
        for k in range(<int>start, <int>end):
            d = 0.
            for j in range(self.nbvar):
                d += self.sample_f[idx] * pC[j]
                idx += 1
            d -= self.sample_y[k]
            squ += d * d * self.sample_w[k]
        return 0. if weight == 0. else squ / weight

    cdef void _node_beta(self, float64_t* dest) noexcept nogil:
        """
        Stores the results of the linear regression
        in an allocated numpy array.

        :param dest: allocated float64_t pointer, size must be *>= self.nbvar*
        """
        self._reglin(self.start, self.end, 1)
        memcpy(dest, self.sample_pC, self.nbvar * sizeof(float64_t))

    def node_beta(self, float64_t[::1] dest):
        """
        Stores the results of the linear regression
        in an allocated numpy array.

        :param dest: allocated array
        """
        if dest.shape[0] < self.nbvar:
            raise ValueError("dest must be at least (%d, )" % self.nbvar)
        self._node_beta(&dest[0])
