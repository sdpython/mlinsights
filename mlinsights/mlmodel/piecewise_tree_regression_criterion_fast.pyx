"""
@file
@brief Implements a custom criterion to train a decision tree.
"""
cimport cython
import numpy
cimport numpy

numpy.import_array()

from libc.stdlib cimport calloc, free
from libc.math cimport NAN

from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t
from ._piecewise_tree_regression_common cimport CommonRegressorCriterion


cdef class SimpleRegressorCriterionFast(CommonRegressorCriterion):
    """
    Criterion which computes the mean square error
    assuming points falling into one node are approximated
    by a constant. The implementation follows the same
    design used in :class:`SimpleRegressorCriterion
    <mlinsights.mlmodel.piecewise_tree_regression_criterion.SimpleRegressorCriterion>`.
    This implementation is faster as it computes
    cumulated sums and avoids loops to compute
    intermediate gains.
    """
    cdef DOUBLE_t* sample_w_left
    cdef DOUBLE_t* sample_wy2_left
    cdef DOUBLE_t* sample_wy_left

    def __dealloc__(self):
        """Destructor."""        
        free(self.sample_w_left)
        free(self.sample_wy_left)
        free(self.sample_wy2_left)
        self.sample_w_left = NULL
        self.sample_wy_left = NULL
        self.sample_wy2_left = NULL

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __cinit__(self, const DOUBLE_t[:, ::1] X):
        self.sample_X = X
        # Allocate memory for the accumulators
        self.sample_w_left = NULL
        self.sample_wy_left = NULL
        self.sample_wy2_left = NULL
        
        # Criterion interface
        self.sample_weight = NULL
        self.samples = NULL
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        
        # allocations
        if self.sample_w_left == NULL:
            self.sample_w_left = <DOUBLE_t*> calloc(X.shape[0], sizeof(DOUBLE_t))
        if self.sample_wy_left == NULL:
            self.sample_wy_left = <DOUBLE_t*> calloc(X.shape[0], sizeof(DOUBLE_t))
        if self.sample_wy2_left == NULL:
            self.sample_wy2_left = <DOUBLE_t*> calloc(X.shape[0], sizeof(DOUBLE_t))

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, 
                  SIZE_t start, SIZE_t end) nogil except -1:
        """
        This function is overwritten to check *y* and *X* size are the same.
        This API has changed in 0.21.
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
        Returns -1 in case of failure to allocate memory
        (and raise *MemoryError*) or 0 otherwise.

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
        cdef SIZE_t ki, ks
        cdef double w, y_

        self.start = start
        self.pos = start
        self.end = end
        self.weighted_n_samples = weighted_n_samples
        self.y = y            

        # we need to do that in case start > 0 or end < X.shape[0]
        for i in range(0, X.shape[0]):
            self.sample_w_left[i] = 0
            self.sample_wy_left[i] = 0
            self.sample_wy2_left[i] = 0

        # Left side.
        for ki in range(<int>start, <int>start+1):
            ks = samples[ki]
            w = sample_weight[ks] if sample_weight else 1.
            y_ = y[ks, 0]
            self.sample_w_left[ki] = w
            self.sample_wy_left[ki] = w * y_
            self.sample_wy2_left[ki] = w * y_ * y_
        for ki in range(<int>start+1, <int>end):
            ks = samples[ki]
            w = sample_weight[ks] if sample_weight else 1.
            y_ = y[ks, 0]
            self.sample_w_left[ki] = self.sample_w_left[ki-1] + w 
            self.sample_wy_left[ki] = self.sample_wy_left[ki-1] + w * y_
            self.sample_wy2_left[ki] = self.sample_wy2_left[ki-1] + w * y_ * y_
        
        self.weighted_n_node_samples = self.sample_w_left[end-1]
        self.reset()
        if self.weighted_n_node_samples == 0:
            raise ValueError(
                "self.weighted_n_node_samples is null, first weight is %r." % self.sample_w[0])
        return 0

    cdef void _mean(self, SIZE_t start, SIZE_t end, DOUBLE_t *mean, DOUBLE_t *weight) nogil:
        """
        Computes the mean of *y* between *start* and *end*.
        """
        if start == end:
            mean[0] = 0.
            return
        cdef DOUBLE_t m = self.sample_wy_left[end-1] - (self.sample_wy_left[start-1] if start > 0 else 0)
        cdef DOUBLE_t w = self.sample_w_left[end-1] - (self.sample_w_left[start-1] if start > 0 else 0)
        weight[0] = w
        mean[0] = 0. if w == 0. else m / w

    cdef double _mse(self, SIZE_t start, SIZE_t end, DOUBLE_t mean, DOUBLE_t weight) nogil:
        """
        Computes mean square error between *start* and *end*
        assuming corresponding points are approximated by a constant.
        """
        if start == end:
            return 0.
        cdef DOUBLE_t squ = self.sample_wy2_left[end-1] - (self.sample_wy2_left[start-1] if start > 0 else 0)
        # This formula only holds if mean is computed on the same interval.
        # Otherwise, it is squ / weight - true_mean ** 2 + (mean - true_mean) ** 2.
        return 0. if weight == 0. else squ / weight - mean ** 2

    cdef void _update_weights(self, SIZE_t start, SIZE_t end, SIZE_t old_pos, SIZE_t new_pos) nogil:
        """
        Updates members `weighted_n_right` and `weighted_n_left`
        when `pos` changes.
        """
        if new_pos == 0:
            self.weighted_n_left = 0.
            self.weighted_n_right = self.sample_w_left[end - 1]
        else:
            self.weighted_n_left = self.sample_w_left[new_pos - 1]
            self.weighted_n_right = self.sample_w_left[end - 1] - self.sample_w_left[new_pos - 1]
