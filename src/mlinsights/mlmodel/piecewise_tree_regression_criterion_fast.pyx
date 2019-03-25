"""
@file
@brief Implements a custom criterion to train a decision tree.
"""
from libc.stdlib cimport calloc, free
from libc.math cimport NAN

import numpy
cimport numpy
numpy.import_array()

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t


cdef class SimpleRegressorCriterionFast(Criterion):
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
    cdef const DOUBLE_t[:, ::1] sample_X

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

        Parameters
        ----------
        X : array-like, features, dtype=DOUBLE_t
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples : DOUBLE_t
            The total weight of the samples being considered
        samples : array-like, dtype=DOUBLE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """
        cdef int ki, ks
        cdef double w, y_

        if y.shape[1] != 1:
            raise ValueError("This class only works for a single vector.")

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
        for ki in range(start, start+1):
            ks = samples[ki]
            w = sample_weight[ks] if sample_weight else 1.
            y_ = y[ks, 0]
            self.sample_w_left[ki] = w
            self.sample_wy_left[ki] = w * y_
            self.sample_wy2_left[ki] = w * y_ * y_
        for ki in range(start+1, end):
            ks = samples[ki]
            w = sample_weight[ks] if sample_weight else 1.
            y_ = y[ks, 0]
            self.sample_w_left[ki] = self.sample_w_left[ki-1] + w 
            self.sample_wy_left[ki] = self.sample_wy_left[ki-1] + w * y_
            self.sample_wy2_left[ki] = self.sample_wy2_left[ki-1] + w * y_ * y_
        
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """
        Resets the criterion at *pos=start*.
        This method must be implemented by the subclass.
        """
        self.pos = self.start

    cdef int reverse_reset(self) nogil except -1:
        """
        Resets the criterion at *pos=end*.
        This method must be implemented by the subclass.
        """
        self.pos = self.end

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """
        Updates statistics by moving ``samples[pos:new_pos]`` to the left child.
        This updates the collected statistics by moving ``samples[pos:new_pos]``
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """
        self.pos = new_pos

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
            
    cdef void children_impurity_weights(self, double* impurity_left,
                                        double* impurity_right,
                                        double* weight_left,
                                        double* weight_right) nogil:
        """
        Calculates the impurity of children.

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored.
        weight_left : double pointer
            The memory address where the weight of the left child should be
            stored.
        weight_right : double pointer
            The memory address where the weight of the right child should be
            stored.
        """
        cdef DOUBLE_t mleft, mright
        self._mean(self.start, self.pos, &mleft, weight_left)
        self._mean(self.pos, self.end, &mright, weight_right)
        impurity_left[0] = self._mse(self.start, self.pos, mleft, weight_left[0])
        impurity_right[0] = self._mse(self.pos, self.end, mright, weight_right[0])

    ####################
    # functions used by a the tree optimizer
    ####################

    cdef double node_impurity(self) nogil:
        """
        Calculates the impurity of the node.
        """
        cdef DOUBLE_t mean, weight
        self._mean(self.start, self.end, &mean, &weight)
        return self._mse(self.start, self.end, mean, weight)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """
        Calculates the impurity of children.
        
        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored.
        """
        cdef DOUBLE_t wl, wr
        self.children_impurity_weights(impurity_left, impurity_right, &wl, &wr)

    cdef void node_value(self, double* dest) nogil:
        """
        Computes the node value, usually the prediction
        the tree would make at this node.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """
        cdef DOUBLE_t weight
        self._mean(self.start, self.end, dest, &weight)

    cdef double proxy_impurity_improvement(self) nogil:
        """
        Computes a proxy of the impurity reduction
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        *impurity_improvement* method once the best split has been found.
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity_weights(&impurity_left, &impurity_right,
                                       &self.weighted_n_left, &self.weighted_n_right)
        if self.pos == self.start or self.pos == self.end:
            return NAN

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity) nogil:
        """
        Computes the improvement in impurity
        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following::
        
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where *N* is the total number of samples, *N_t* is the number of samples
        at the current node, *N_t_L* is the number of samples in the left child,
        and *N_t_R* is the number of samples in the right child,

        Parameters
        ----------
        impurity : double
            The initial impurity of the node before the split
        Return
        ------
        double : improvement in impurity after the split occurs
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity_weights(&impurity_left, &impurity_right,
                                       &self.weighted_n_left, &self.weighted_n_right)
        if self.pos == self.start or self.pos == self.end:
            return NAN

        cdef double weight = self.weighted_n_left + self.weighted_n_right
        return ((weight / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right / weight * impurity_right)
                          - (self.weighted_n_left / weight * impurity_left)))
