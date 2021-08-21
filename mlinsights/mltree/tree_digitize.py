"""
@file
@brief Helpers to investigate a tree structure.
"""
import numpy
from sklearn.tree._tree import Tree  # pylint: disable=E0611
from sklearn.tree import DecisionTreeRegressor
from ._tree_digitize import tree_add_node


def digitize2tree(bins, right=False):
    """
    Builds a decision tree which returns the same result as
    `lambda x: numpy.digitize(x, bins, right=right)`
    (see :epkg:`numpy:digitize`).

    :param bins: array of bins. It has to be 1-dimensional and monotonic.
    :param right: Indicating whether the intervals include the right
        or the left bin edge. Default behavior is (right==False)
        indicating that the interval does not include the right edge.
        The left bin end is open in this case, i.e.,
        `bins[i-1] <= x < bins[i]` is the default behavior for
        monotonically increasing bins.
    :return: decision tree
    """
    if not right:
        raise NotImplementedError("right must be true")
    tree = Tree(1, numpy.array([1], dtype=numpy.intp), 1)
    ascending = len(bins) <= 1 or bins[0] < bins[1]
    if not ascending:
        raise NotImplementedError("ascending must be true")
    values = []

    def add_root(index):
        if index < 0 or index >= len(bins):
            raise IndexError(  # pragma: no cover
                "Unexpected index %d / len(bins)=%d." % (
                    index, len(bins)))
        parent = -1
        is_left = False
        is_leaf = False
        threshold = bins[index]
        tree_add_node(tree, parent, is_left, is_leaf, 0, threshold, 0, 1, 1.)
        values.append(numpy.nan)

    def add_nodes(parent, i, j, is_left):
        # add for bins[i:j] (j excluded)
        if i == j == 0:
            # leaf
            tree_add_node(tree, parent, is_left, True, 0, 0, 0, 1, 1.)
            values.append(0)
        elif i + 1 == j:
            # leaf
            threshold = j if ascending else i
            values.append(threshold)
            tree_add_node(tree, parent, is_left, True, 0, 0, 0, 1, 1.)
        else:
            raise NotImplementedError("i=%r j=%r" % (i, j))

    index = len(bins) // 2
    add_root(index)
    add_nodes(0, 0, index, True)
    add_nodes(0, index, len(bins), False)

    cl = DecisionTreeRegressor()
    cl.tree_ = tree
    cl.tree_.value[:, 0, 0] = numpy.array(values, dtype=numpy.float64)
    cl.n_outputs = 1
    cl.n_outputs_ = 1
    cl.n_features_in_ = 1
    return cl
