# See _piecewise_tree_regression_common.pyx for implementation details.
import numpy
cimport numpy
cimport cython

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t


cdef class CommonRegressorCriterion(Criterion):

    cdef public const DOUBLE_t[:, ::1] sample_X

    cdef void _mean(self, SIZE_t start, SIZE_t end, DOUBLE_t *mean,
                    DOUBLE_t *weight) nogil
    cdef double _mse(self, SIZE_t start, SIZE_t end, DOUBLE_t mean,
                     DOUBLE_t weight) nogil
