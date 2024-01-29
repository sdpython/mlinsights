from libc.math cimport NAN
cimport numpy as cnp

cnp.import_array()

from sklearn.tree._criterion cimport Criterion


cdef class CommonRegressorCriterion(Criterion):
    """
    Common class to implement various version of `mean square error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_.
    The code was inspired from
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
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __deepcopy__(self, memo=None):
        """
        This does not a copy but mostly creates a new instance
        of the same criterion initialized with the same data.
        """
        inst = self.__class__(self.n_outputs, self.n_samples)
        return inst

    cdef void _update_weights(self, intp_t start, intp_t end,
                              intp_t old_pos, intp_t new_pos) noexcept nogil:
        """
        Updates members `weighted_n_right` and `weighted_n_left`
        when `pos` changes. This method should be overloaded.
        """
        pass

    cdef int reset(self) except -1 nogil:
        """
        Resets the criterion at *pos=start*.
        This method must be implemented by the subclass.
        """
        self._update_weights(self.start, self.end, self.pos, self.start)
        self.pos = self.start

    cdef int reverse_reset(self) except -1 nogil:
        """
        Resets the criterion at *pos=end*.
        This method must be implemented by the subclass.
        """
        self._update_weights(self.start, self.end, self.pos, self.end)
        self.pos = self.end

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """
        Updates statistics by moving ``samples[pos:new_pos]`` to the left child.
        This updates the collected statistics by moving ``samples[pos:new_pos]``
        from the right child to the left child. It must be implemented by
        the subclass.

        :param new_pos: intp_t
            New starting index position of the samples in the right child
        """
        self._update_weights(self.start, self.end, self.pos, new_pos)
        self.pos = new_pos

    cdef void _mean(self, intp_t start, intp_t end, float64_t *mean,
                    float64_t *weight) noexcept nogil:
        """
        Computes the mean of *y* between *start* and *end*.
        """
        pass

    cdef float64_t _mse(self, intp_t start, intp_t end, float64_t mean,
                        float64_t weight) noexcept nogil:
        """
        Computes mean square error between *start* and *end*
        assuming corresponding points are approximated by a constant.
        """
        return 0.0

    cdef void children_impurity_weights(self, float64_t* impurity_left,
                                        float64_t* impurity_right,
                                        float64_t* weight_left,
                                        float64_t* weight_right) noexcept nogil:
        """
        Calculates the impurity of children,
        evaluates the impurity in
        children nodes, i.e. the impurity of ``samples[start:pos]``
        the impurity of ``samples[pos:end]``.

        :param impurity_left: float64_t pointer
            The memory address where the impurity of the left child should be
            stored.
        :param impurity_right: float64_t pointer
            The memory address where the impurity of the right child should be
            stored.
        :param weight_left: float64_t pointer
            The memory address where the weight of the left child should be
            stored.
        :param weight_right: float64_t pointer
            The memory address where the weight of the right child should be
            stored.
        """
        cdef float64_t mleft, mright
        self._mean(self.start, self.pos, &mleft, weight_left)
        self._mean(self.pos, self.end, &mright, weight_right)
        impurity_left[0] = self._mse(self.start, self.pos, mleft, weight_left[0])
        impurity_right[0] = self._mse(self.pos, self.end, mright, weight_right[0])

    ####################
    # functions used by a the tree optimizer
    ####################

    cdef float64_t node_impurity(self) noexcept nogil:
        """
        Calculates the impurity of the node,
        the impurity of ``samples[start:end]``.
        This is the primary function of the criterion class.
        """
        cdef float64_t mean, weight
        self._mean(self.start, self.end, &mean, &weight)
        return self._mse(self.start, self.end, mean, weight)

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """
        Calculates the impurity of children.

        :param impurity_left: float64_t pointer
            The memory address where the impurity of the left child should be
            stored.
        :param impurity_right: float64_t pointer
            The memory address where the impurity of the right child should be
            stored.
        """
        cdef float64_t wl, wr
        self.children_impurity_weights(impurity_left, impurity_right, &wl, &wr)

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """
        Computes the node value, usually, the prediction
        the tree would do. Stores the value into *dest*.

        :param dest: float64_t pointer
            The memory address where the node value should be stored.
        """
        cdef float64_t weight
        self._mean(self.start, self.end, dest, &weight)

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """
        Computes a proxy of the impurity reduction
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        *impurity_improvement* method once the best split has been found.
        """
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        self.children_impurity_weights(&impurity_left, &impurity_right,
                                       &self.weighted_n_left, &self.weighted_n_right)
        if self.pos == self.start or self.pos == self.end:
            return NAN

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        """
        Computes the improvement in impurity
        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where *N* is the total number of samples, *N_t* is the number of samples
        at the current node, *N_t_L* is the number of samples in the left child,
        and *N_t_R* is the number of samples in the right child,

        :param impurity_parent: float64_t
            The initial impurity of the node before the split
        :param impurity_left: float64_t
            The impurity of the left child
        :param impurity_right: float64_t
            The impurity of the right child
        :return: float64_t, improvement in impurity after the split occurs
        """
        # self.children_impurity_weights(&impurity_left, &impurity_right,
        #                                &self.weighted_n_left, &self.weighted_n_right)
        # if self.pos == self.start or self.pos == self.end:
        #     return NAN

        # cdef float64_t weight = self.weighted_n_left + self.weighted_n_right
        cdef float64_t weight = self.weighted_n_node_samples
        return ((weight / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right / weight * impurity_right)
                                 - (self.weighted_n_left / weight * impurity_left)))


cdef int _ctest_criterion_init(Criterion criterion,
                               const float64_t[:, ::1] y,
                               float64_t[:] sample_weight,
                               float64_t weighted_n_samples,
                               const intp_t[:] samples,
                               intp_t start, intp_t end):
    "Test purposes. Methods cannot be directly called from python."
    cdef const float64_t[:, ::1] y2 = y
    return criterion.init(y2, sample_weight, weighted_n_samples, samples, start, end)


def _test_criterion_init(Criterion criterion,
                         const float64_t[:, ::1] y,
                         float64_t[:] sample_weight,
                         float64_t weighted_n_samples,
                         const intp_t[:] samples,
                         intp_t start, intp_t end):
    "Test purposes. Methods cannot be directly called from python."
    if _ctest_criterion_init(criterion, y, sample_weight, weighted_n_samples,
                             samples, start, end) != 0:
        raise AssertionError("Return is not 0.")


def _test_criterion_check(Criterion criterion):
    if criterion.weighted_n_node_samples == 0:
        raise ValueError(
            f"weighted_n_node_samples is null, "
            f"weighted_n_left={criterion.weighted_n_left!r}, "
            f"weighted_n_right={criterion.weighted_n_right}"
        )


def assert_criterion_equal(Criterion c1, Criterion c2):
    if c1.weighted_n_node_samples != c2.weighted_n_node_samples:
        raise ValueError(
            "weighted_n_node_samples: %r != %r" % (
                c1.weighted_n_node_samples, c2.weighted_n_node_samples))
    if c1.weighted_n_samples != c2.weighted_n_samples:
        raise ValueError(
            "weighted_n_samples: %r != %r" % (
                c1.weighted_n_samples, c2.weighted_n_samples))
    if c1.weighted_n_left != c2.weighted_n_left:
        raise ValueError(
            "weighted_n_left: %r != %r" % (
                c1.weighted_n_left, c2.weighted_n_left))
    if c1.weighted_n_right != c2.weighted_n_right:
        raise ValueError(
            "weighted_n_right: %r != %r" % (
                c1.weighted_n_right, c2.weighted_n_right))


def _test_criterion_node_impurity(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.node_impurity()


def _test_criterion_proxy_impurity_improvement(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.proxy_impurity_improvement()


def _test_criterion_impurity_improvement(Criterion criterion,
                                         float64_t impurity_parent,
                                         float64_t impurity_left,
                                         float64_t impurity_right):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.impurity_improvement(
        impurity_parent, impurity_left, impurity_right
    )


def _test_criterion_node_impurity_children(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    cdef float64_t left, right
    criterion.children_impurity(&left, &right)
    return left, right


def _test_criterion_node_value(Criterion criterion):
    "Test purposes. Methods cannot be directly called from python."
    cdef float64_t value
    criterion.node_value(&value)
    return value


def _test_criterion_update(Criterion criterion, intp_t new_pos):
    "Test purposes. Methods cannot be directly called from python."
    return criterion.update(new_pos)


def _test_criterion_printf(Criterion crit):
    "Test purposes. Methods cannot be directly called from python."
    cdef float64_t left, right, value
    cdef int i
    crit.children_impurity(&left, &right)
    crit.node_value(&value)
    rows = []
    rows.append("start=%d pos=%d end=%d" % (crit.start, crit.pos, crit.end))
    rows.append("value: %f total=%f left=%f right=%f" % (
            value, crit.node_impurity(), left, right))
    cdef int n = crit.y.shape[0]
    for i in range(0, n):
        rows.append("-- %d: y=%f" % (i, crit.y[i, 0]))
    return "\n".join(rows)
