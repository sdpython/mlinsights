# See _piecewise_tree_regression_common.pyx for implementation details.
cimport cython
import numpy
cimport numpy as cnp

cnp.import_array()

from sklearn.tree._criterion cimport Criterion

ctypedef double float64_t
ctypedef cnp.npy_intp SIZE_t


cdef class CommonRegressorCriterion(Criterion):

    cdef void _update_weights(self, SIZE_t start, SIZE_t end,
                              SIZE_t old_pos, SIZE_t new_pos) noexcept nogil

    cdef void _mean(self, SIZE_t start, SIZE_t end,
                    float64_t *mean, float64_t *weight) noexcept nogil

    cdef float64_t _mse(self, SIZE_t start, SIZE_t end,
                        float64_t mean, float64_t weight) noexcept nogil

    cdef void children_impurity_weights(self, float64_t* impurity_left,
                                        float64_t* impurity_right,
                                        float64_t* weight_left,
                                        float64_t* weight_right) noexcept nogil
