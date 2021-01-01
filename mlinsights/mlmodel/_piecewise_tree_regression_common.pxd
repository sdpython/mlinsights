# See _piecewise_tree_regression_common.pyx for implementation details.
cimport cython
import numpy
cimport numpy

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t


cdef class CommonRegressorCriterion(Criterion):

    cdef const DOUBLE_t[:, ::1] sample_X

    cdef void _update_weights(self, SIZE_t start, SIZE_t end,
                              SIZE_t old_pos, SIZE_t new_pos) nogil

    cdef void _mean(self, SIZE_t start, SIZE_t end, DOUBLE_t *mean,
                    DOUBLE_t *weight) nogil
    cdef double _mse(self, SIZE_t start, SIZE_t end, DOUBLE_t mean,
                     DOUBLE_t weight) nogil

    cdef void children_impurity_weights(self, double* impurity_left,
                                        double* impurity_right,
                                        double* weight_left,
                                        double* weight_right) nogil
