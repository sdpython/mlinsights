"""
@file
@brief Implements a custom criterion to train a decision tree.
"""
from libc.stdlib cimport calloc, free
from libc.stdio cimport printf
from libc.math cimport NAN

import numpy
cimport numpy
numpy.import_array()

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._criterion cimport SIZE_t, DOUBLE_t


cdef class CommonRegressorCriterion(Criterion):
    """
    Common class to implement various version of `mean square error 
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_.
    The code was inspired from
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

    If the file does not compile, some explanations are given
    in :ref:`scikit-learn internal API
    <blog-internal-api-impurity-improvement>`_.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __cinit__(self, const DOUBLE_t[:, ::1] X):
        self.sample_X = X

    cdef void _update_weights(self, SIZE_t start, SIZE_t end, SIZE_t old_pos, SIZE_t new_pos) nogil:
        """
        Unused.
        """
        pass

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

        :param new_pos: SIZE_t
            New starting index position of the samples in the right child
        """
        self.pos = new_pos

    cdef void _mean(self, SIZE_t start, SIZE_t end, DOUBLE_t *mean, DOUBLE_t *weight) nogil:
        """
        Computes the mean of *y* between *start* and *end*.
        """
        raise NotImplementedError("Method _mean must be overloaded.")
            
    cdef double _mse(self, SIZE_t start, SIZE_t end, DOUBLE_t mean, DOUBLE_t weight) nogil:
        """
        Computes mean square error between *start* and *end*
        assuming corresponding points are approximated by a constant.
        """
        raise NotImplementedError("Method _mean must be overloaded.")
            
    cdef void children_impurity_weights(self, double* impurity_left,
                                        double* impurity_right,
                                        double* weight_left,
                                        double* weight_right) nogil:
        """
        Calculates the impurity of children,
        evaluates the impurity in
        children nodes, i.e. the impurity of ``samples[start:pos]``
        the impurity of ``samples[pos:end]``.

        :param impurity_left: double pointer
            The memory address where the impurity of the left child should be
            stored.
        :param impurity_right: double pointer
            The memory address where the impurity of the right child should be
            stored.
        :param weight_left: double pointer
            The memory address where the weight of the left child should be
            stored.
        :param weight_right: double pointer
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
        Calculates the impurity of the node,
        the impurity of ``samples[start:end]``.
        This is the primary function of the criterion class.
        """
        cdef DOUBLE_t mean, weight
        self._mean(self.start, self.end, &mean, &weight)
        return self._mse(self.start, self.end, mean, weight)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """
        Calculates the impurity of children.
        
        :param impurity_left: double pointer
            The memory address where the impurity of the left child should be
            stored.
        :param impurity_right: double pointer
            The memory address where the impurity of the right child should be
            stored.
        """
        cdef DOUBLE_t wl, wr
        self.children_impurity_weights(impurity_left, impurity_right, &wl, &wr)

    cdef void node_value(self, double* dest) nogil:
        """
        Computes the node value, usually, the prediction
        the tree would do. Stores the value into *dest*.

        :param dest: double pointer
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

        :param impurity: double
            The initial impurity of the node before the split
        :return: double, improvement in impurity after the split occurs
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


def _test_criterion_init(Criterion criterion, 
                        const DOUBLE_t[:, ::1] y,
                        DOUBLE_t[:] sample_weight,
                        double weighted_n_samples,
                        SIZE_t[:] samples, 
                        SIZE_t start, SIZE_t end):
    "Test purposes. Methods cannot be directly called from python."
    criterion.init(y,
                   &sample_weight[0], weighted_n_samples,
                   &samples[0], start, end)


def _test_criterion_check(Criterion criterion):
    "Unused"
    pass


def assert_criterion_equal(Criterion c1, Criterion c2):
    "Unused"
    pass


def _test_criterion_node_impurity(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.node_impurity()

    
def _test_criterion_proxy_impurity_improvement(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.proxy_impurity_improvement()

    
def _test_criterion_impurity_improvement(Criterion criterion, double impurity):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.impurity_improvement(impurity)

    
def _test_criterion_node_impurity_children(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    cdef DOUBLE_t left, right
    criterion.children_impurity(&left, &right)
    return left, right


def _test_criterion_node_value(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    cdef DOUBLE_t value
    criterion.node_value(&value)
    return value


def _test_criterion_update(Criterion criterion, SIZE_t new_pos):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.update(new_pos)

    
def _test_criterion_printf(Criterion crit):
    "Test purposes. Methods cannot be directly called from python."    
    printf("start=%zu pos=%zu end=%zu\n", crit.start, crit.pos, crit.end)
    cdef DOUBLE_t left, right, value
    cdef int i;
    crit.children_impurity(&left, &right)
    crit.node_value(&value)
    printf("value: %f total=%f left=%f right=%f\n", value, 
           crit.node_impurity(), left, right)
    cdef int n = crit.y.shape[0]
    for i in range(0, n):
        printf("-- %d: y=%f\n", i, crit.y[i, 0])
