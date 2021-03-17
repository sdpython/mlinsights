"""
@file
@brief Implements a custom criterion to train a decision tree.
"""
cimport cython
import numpy
cimport numpy

numpy.import_array()

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.math cimport NAN

cimport scipy.linalg.cython_lapack as cython_lapack
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t
from ._piecewise_tree_regression_common cimport CommonRegressorCriterion


cdef class LinearRegressorCriterion(CommonRegressorCriterion):
    """
    Criterion which computes the mean square error
    assuming points falling into one node are approximated
    by a line (linear regression).
    The implementation follows the same
    design used in :class:`SimpleRegressorCriterion
    <mlinsights.mlmodel.piecewise_tree_regression_criterion.SimpleRegressorCriterion>`
    and is even slow as the criterion is more complex to compute.
    """
    cdef DOUBLE_t* sample_w
    cdef DOUBLE_t* sample_y
    cdef DOUBLE_t* sample_wy
    cdef DOUBLE_t* sample_f
    cdef DOUBLE_t* sample_pC
    cdef DOUBLE_t* sample_pS
    cdef DOUBLE_t* sample_work
    cdef SIZE_t* sample_i
    cdef DOUBLE_t* sample_f_buffer
    
    cdef DOUBLE_t sample_sum_wy
    cdef DOUBLE_t sample_sum_w
    cdef SIZE_t nbvar
    cdef SIZE_t nbrows
    cdef SIZE_t work

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

    def __cinit__(self, const DOUBLE_t[:, ::1] X):
        self.sample_X = X
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
        self.sample_weight = NULL
        self.samples = NULL
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL            

        # allocation
        if self.sample_w == NULL:
            self.sample_w = <DOUBLE_t*> calloc(X.shape[0], sizeof(DOUBLE_t))
        if self.sample_wy == NULL:
            self.sample_wy = <DOUBLE_t*> calloc(X.shape[0], sizeof(DOUBLE_t))
        if self.sample_y == NULL:
            self.sample_y = <DOUBLE_t*> calloc(X.shape[0], sizeof(DOUBLE_t))
        if self.sample_i == NULL:
            self.sample_i = <SIZE_t*> calloc(X.shape[0], sizeof(SIZE_t))
        if self.sample_f == NULL:
            self.sample_f = <DOUBLE_t*> calloc(X.shape[0] * (X.shape[1] + 1), sizeof(DOUBLE_t))
            
        self.nbvar = X.shape[1] + 1
        self.nbrows = X.shape[0]
        self.work = <SIZE_t>(min(self.nbrows, self.nbvar) * <SIZE_t>3 + 
                             max(max(self.nbrows, self.nbvar),
                                 min(self.nbrows, self.nbvar) * <SIZE_t>2))
        if self.sample_f_buffer == NULL:
            self.sample_f_buffer = <DOUBLE_t*> calloc(X.shape[0] * self.nbvar, sizeof(DOUBLE_t))
        if self.sample_pC == NULL:
            self.sample_pC = <DOUBLE_t*> calloc(max(self.nbrows, self.nbvar), sizeof(DOUBLE_t))
        if self.sample_work == NULL:
            self.sample_work = <DOUBLE_t*> calloc(self.work, sizeof(DOUBLE_t))
        if self.sample_pS == NULL:
            self.sample_pS = <DOUBLE_t*> calloc(self.nbvar, sizeof(DOUBLE_t))

    @staticmethod
    def create(DOUBLE_t[:, ::1] X, DOUBLE_t[:, ::1] y, DOUBLE_t[::1] sample_weight=None):
        """
        Initializes the criterion.
        
        :param X: features
        :param y: target
        :param sample_weight: sample weight
        :return: an instance of :class:`LinearRegressorCriterion`
        """
        cdef SIZE_t i
        cdef DOUBLE_t* ws
        cdef double sum
        cdef SIZE_t* parr = <SIZE_t*> calloc(y.shape[0], sizeof(SIZE_t))
        for i in range(0, y.shape[0]):
            parr[i] = i
        if sample_weight is None:
            sum = <DOUBLE_t>y.shape[0]
            ws = NULL
        else:
            sum = sample_weight.sum()
            ws = &sample_weight[0]

        obj = LinearRegressorCriterion(X)
        obj.init(y, ws, sum, parr, 0, y.shape[0])            
        free(parr)
        return obj

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, 
                  SIZE_t start, SIZE_t end) nogil except -1:
        """
        This function is overwritten to check *y* and *X* size are the same.
        This API changed in 0.21.
        """
        if y.shape[0] != self.sample_X.shape[0]:
            raise ValueError("X.shape={} -- y.shape={}".format(self.sample_X.shape, y.shape))
        if y.shape[1] != 1:
            raise ValueError("This class only works for a single vector.")
        return self.init_with_X(self.sample_X, y, sample_weight, weighted_n_samples,
                                samples, start, end)

    cdef int init_with_X(self, const DOUBLE_t[:, ::1] X, 
                         const DOUBLE_t[:, ::1] y,
                         DOUBLE_t* sample_weight,
                         double weighted_n_samples, SIZE_t* samples, 
                         SIZE_t start, SIZE_t end) nogil except -1:
        """
        Initializes the criterion.

        :param X: array-like, features, dtype=DOUBLE_t
        :param y: array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        :param sample_weight: array-like, dtype=DOUBLE_t
            The weight of each sample
        :param weighted_n_samples: DOUBLE_t
            The total weight of the samples being considered
        :param samples: array-like, dtype=DOUBLE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        :param start: SIZE_t
            The first sample to be used on this node
        :param end: SIZE_t
            The last sample used on this node
        """
        cdef SIZE_t ki, ks, idx, c

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
            ks = samples[ki]
            self.sample_i[ki] = ks
            self.sample_w[ki] = sample_weight[ks] if sample_weight else 1.
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
                "self.weighted_n_node_samples is null, first weight is %r." % self.sample_w[0])
        return 0

    cdef void _mean(self, SIZE_t start, SIZE_t end, DOUBLE_t *mean, DOUBLE_t *weight) nogil:
        """
        Computes mean between *start* and *end*.
        """
        if start == end:
            mean[0] = 0.
            return
        cdef DOUBLE_t m = 0.
        cdef DOUBLE_t w = 0.
        cdef SIZE_t k
        for k in range(<int>start, <int>end):
            m += self.sample_wy[k]
            w += self.sample_w[k]
        weight[0] = w
        mean[0] = 0. if w == 0. else m / w
            
    cdef void _reglin(self, SIZE_t start, SIZE_t end, int low_rank) nogil:
        """
        Solves the linear regression between *start* and *end*
        assuming corresponding points are approximated by a line.
        *mean* is unused but could be if the inverse does not exist.
        The solution is the vector ``self.sample_pC[:self.nbvar]``.
        """
        cdef SIZE_t i, j, idx, pos
        cdef DOUBLE_t w
        cdef DOUBLE_t* sample_f_buffer = self.sample_f_buffer
        pos = 0
        for j in range(self.nbvar):
            idx = start * self.nbvar + j
            for i in range(<int>start, <int>end):
                w = self.sample_w[i]
                sample_f_buffer[pos] = self.sample_f[idx] * w
                idx += self.nbvar
                pos += 1
        
        cdef DOUBLE_t* pC = self.sample_pC
        for i in range(<int>start, <int>end):
            pC[i-start] = self.sample_wy[i]

        cdef int col = <int>self.nbvar
        cdef int row = <int>(end - start)
        cdef int info
        cdef int nrhs = 1
        cdef int lda = row
        cdef int ldb = row
        cdef DOUBLE_t rcond = -1
        cdef int rank        
        cdef int work = <int>self.work
        
        if row < col:
            if low_rank:
                ldb = col
            else:
                raise RuntimeError("The function cannot return any return when row < col.")
        cython_lapack.dgelss(&row, &col, &nrhs,                 # 1-3
                             sample_f_buffer, &lda, pC, &ldb,   # 4-7
                             self.sample_pS, &rcond, &rank,     # 8-10
                             self.sample_work, &work, &info)    # 11-13
                             
    cdef double _mse(self, SIZE_t start, SIZE_t end, DOUBLE_t mean, DOUBLE_t weight) nogil:
        """
        Computes mean square error between *start* and *end*
        assuming corresponding points are approximated by a line.
        *mean* is unused but could be if the inverse does not exist.
        """
        if end - start <= self.nbvar:
            # More coefficients than the number of observations.
            return 0.
        
        self._reglin(start, end, 0)
        
        cdef double* pC = self.sample_pC
        cdef SIZE_t j, idx
        
        # replaces what follows by gemm
        cdef DOUBLE_t squ = 0.
        cdef DOUBLE_t d
        cdef SIZE_t k
        idx = start * self.nbvar
        for k in range(<int>start, <int>end):
            d = 0.
            for j in range(self.nbvar):
                d += self.sample_f[idx] * pC[j]
                idx += 1
            d -= self.sample_y[k]
            squ += d * d * self.sample_w[k]
        return 0. if weight == 0. else squ / weight
        
    cdef void _node_beta(self, double* dest) nogil:
        """
        Stores the results of the linear regression
        in an allocated numpy array.
        
        :param dest: allocated double pointer, size must be *>= self.nbvar*
        """
        self._reglin(self.start, self.end, 1)
        memcpy(dest, self.sample_pC, self.nbvar * sizeof(double))

    def node_beta(self, double[::1] dest):
        """
        Stores the results of the linear regression
        in an allocated numpy array.
        
        :param dest: allocated array
        """
        if dest.shape[0] < self.nbvar:
            raise ValueError("dest must be at least (%d, )" % self.nbvar)
        self._node_beta(&dest[0])
