"""
@file
@brief Helpers to investigate a tree structure.

.. versionadded:: 0.4
"""
import numpy
from sklearn.tree._tree import Tree  # pylint: disable=E0611
from sklearn.tree import DecisionTreeRegressor
from ._tree_digitize import tree_add_node  # pylint: disable=E0611


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

    .. note::
        The implementation of decision trees in :epkg:`scikit-learn`
        only allows one type of decision (`<=`). That's why the
        function throws an exception when `right=False`. However,
        this could be overcome by using :epkg:`ONNX` where all
        kind of decision rules are implemented. Default value for
        right is still *False* to follow *numpy* API even though
        this value raises an exception in *digitize2tree*.

    The following example shows what the tree looks like.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.tree import export_text
        from mlinsights.mltree import digitize2tree

        x = numpy.array([0.2, 6.4, 3.0, 1.6])
        bins = numpy.array([0.0, 1.0, 2.5, 4.0, 7.0])
        expected = numpy.digitize(x, bins, right=True)
        tree = digitize2tree(bins, right=True)
        pred = tree.predict(x.reshape((-1, 1)))
        print("Comparison with numpy:")
        print(expected, pred)
        print("Tree:")
        print(export_text(tree, feature_names=['x']))

    .. versionadded:: 0.4
    """
    if not right:
        raise RuntimeError(
            "right must be True not right=%r" % right)
    ascending = len(bins) <= 1 or bins[0] < bins[1]

    if not ascending:
        bins2 = bins[::-1]
        cl = digitize2tree(bins2, right=right)
        n = len(bins)
        for i in range(cl.tree_.value.shape[0]):
            cl.tree_.value[i, 0, 0] = n - cl.tree_.value[i, 0, 0]
        return cl

    tree = Tree(1, numpy.array([1], dtype=numpy.intp), 1)
    values = []
    UNUSED = numpy.nan
    n_nodes = []

    def add_root(index):
        if index < 0 or index >= len(bins):
            raise IndexError(  # pragma: no cover
                "Unexpected index %d / len(bins)=%d." % (
                    index, len(bins)))
        parent = -1
        is_left = False
        is_leaf = False
        threshold = bins[index]
        n = tree_add_node(
            tree, parent, is_left, is_leaf, 0, threshold, 0, 1, 1.)
        values.append(UNUSED)
        n_nodes.append(n)
        return n

    def add_nodes(parent, i, j, is_left):
        # add for bins[i:j] (j excluded)
        if is_left:
            # it means j is the parent split
            if i == j:
                # leaf
                n = tree_add_node(tree, parent, is_left, True, 0, 0, 0, 1, 1.)
                n_nodes.append(n)
                values.append(i)
                return n
            if i + 1 == j:
                # split
                values.append(UNUSED)
                th = bins[i]
                n = tree_add_node(tree, parent, is_left,
                                  False, 0, th, 0, 1, 1.)
                n_nodes.append(n)
                add_nodes(n, i, i, True)
                add_nodes(n, i, j, False)
                return n
            if i + 1 < j:
                # split
                values.append(UNUSED)
                index = (i + j) // 2
                th = bins[index]
                n = tree_add_node(tree, parent, is_left,
                                  False, 0, th, 0, 1, 1.)
                n_nodes.append(n)
                add_nodes(n, i, index, True)
                add_nodes(n, index, j, False)
                return n
        else:
            # it means i is the parent split
            if i + 1 == j:
                # leaf
                values.append(j)
                n = tree_add_node(tree, parent, is_left, True, 0, 0, 0, 1, 1.)
                n_nodes.append(n)
                return n
            if i + 1 < j:
                # split
                values.append(UNUSED)
                index = (i + j) // 2
                th = bins[index]
                n = tree_add_node(tree, parent, is_left,
                                  False, 0, th, 0, 1, 1.)
                n_nodes.append(n)
                add_nodes(n, i, index, True)
                add_nodes(n, index, j, False)
                return n
        raise NotImplementedError(  # pragma: no cover
            "Unexpected case where i=%r, j=%r, is_left=%r." % (
                i, j, is_left))

    index = len(bins) // 2
    add_root(index)
    add_nodes(0, 0, index, True)
    add_nodes(0, index, len(bins), False)

    cl = DecisionTreeRegressor()
    cl.tree_ = tree
    cl.tree_.value[:, 0, 0] = numpy.array(values, dtype=numpy.float64)
    cl.n_outputs = 1
    cl.n_outputs_ = 1
    try:
        # scikit-learn >= 0.24
        cl.n_features_in_ = 1
    except AttributeError:
        # scikit-learn < 0.24
        cl.n_features_ = 1
    return cl
