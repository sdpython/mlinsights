"""
@file
@brief Implements a base class for a custom criterion to train a decision tree.
"""
cimport cython
import numpy
cimport numpy

numpy.import_array()

from libc.stdlib cimport calloc, free
from libc.math cimport NAN

from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t
from ._piecewise_tree_regression_common cimport CommonRegressorCriterion


cdef class SimpleRegressorCriterion(CommonRegressorCriterion):
    """
    Implements `mean square error 
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    criterion in a non efficient way. The code was inspired from
    `hellinger_distance_criterion.pyx 
    <https://github.com/EvgeniDubov/hellinger-distance-criterion/blob/master/
    hellinger_distance_criterion.pyx>`_,    
    `Cython example of exposing C-computed arrays in Python without data copies
    <http://gael-varoquaux.info/programming/
    cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html>`_,
    `_criterion.pyx
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pyx>`_.
    This implementation is not efficient but was made that way on purpose.
    It adds the features to the class.
    """
    cdef DOUBLE_t* sample_w
    cdef DOUBLE_t* sample_wy
    cdef SIZE_t* sample_i
    cdef DOUBLE_t sample_sum_wy
    cdef DOUBLE_t sample_sum_w

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

    def __cinit__(self, const DOUBLE_t[:, ::1] X):
        self.sample_X = X
        # Allocate memory for the accumulators
        self.sample_w = NULL
        self.sample_wy = NULL
        self.sample_i = NULL
        
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
        if self.sample_i == NULL:
            self.sample_i = <SIZE_t*> calloc(X.shape[0], sizeof(SIZE_t))

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

        self.start = start
        self.pos = start
        self.end = end
        self.weighted_n_samples = weighted_n_samples
        self.y = y            

        self.sample_sum_wy = 0.
        self.sample_sum_w = 0.

        # Filling accumulators.
        for ki in range(<int>start, <int>end):
            ks = samples[ki]
            self.sample_i[ki] = ks
            self.sample_w[ki] = sample_weight[ks] if sample_weight else 1.
            self.sample_wy[ki] = self.sample_w[ki] * y[ks, 0]
            self.sample_sum_wy += y[ks, 0] * self.sample_w[ki]
            self.sample_sum_w += self.sample_w[ki]

        self.weighted_n_node_samples = self.sample_sum_w
        self.reset()
        if self.weighted_n_node_samples == 0:
            raise ValueError(
                "self.weighted_n_node_samples is null, first weight is %r." % self.sample_w[0])
        return 0

    cdef void _update_weights(self, SIZE_t start, SIZE_t end, SIZE_t old_pos, SIZE_t new_pos) nogil:
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

    cdef void _mean(self, SIZE_t start, SIZE_t end, DOUBLE_t *mean, DOUBLE_t *weight) nogil:
        """
        Computes the mean of *y* between *start* and *end*.
        """
        if start == end:
            mean[0] = 0.
            return
        cdef DOUBLE_t m = 0.
        cdef DOUBLE_t w = 0.
        cdef int k
        for k in range(<int>start, <int>end):
            m += self.sample_wy[k]
            w += self.sample_w[k]
        weight[0] = w
        mean[0] = 0. if w == 0. else m / w
            
    cdef double _mse(self, SIZE_t start, SIZE_t end, DOUBLE_t mean, DOUBLE_t weight) nogil:
        """
        Computes mean square error between *start* and *end*
        assuming corresponding points are approximated by a constant.
        """
        if start == end:
            return 0.
        cdef DOUBLE_t squ = 0.
        cdef int k
        for k in range(<int>start, <int>end):            
            squ += (self.y[self.sample_i[k], 0] - mean) ** 2 * self.sample_w[k]
        return 0. if weight == 0. else squ / weight
