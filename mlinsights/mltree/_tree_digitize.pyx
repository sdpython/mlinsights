"""
@file
@brief Access to the C API of scikit-learn (decision tree)
"""
from libc.stdio cimport printf

import numpy
cimport numpy
numpy.import_array()

ctypedef numpy.npy_intp SIZE_t

from sklearn.tree._tree cimport Tree

TREE_LEAF = -1
TREE_UNDEFINED = -2


cdef SIZE_t _tree_add_node(Tree tree,
                           SIZE_t parent,
                           bint is_left,
                           bint is_leaf,
                           SIZE_t feature,
                           double threshold,
                           double impurity,
                           SIZE_t n_node_samples,
                           double weighted_n_node_samples):
    if parent == -1:
        parent = TREE_UNDEFINED
    return tree._add_node(parent, is_left, is_leaf, feature,
                          threshold, impurity,
                          n_node_samples, weighted_n_node_samples)


def tree_add_node(tree, parent, is_left, is_leaf, feature, threshold,
                  impurity, n_node_samples, weighted_n_node_samples):
    """
    Adds a node to tree.

    :param parent: parent index (-1 for the root)
    :param is_left: is left node?
    :param is_leaf: is leave?
    :param feature: feature index
    :param threshold: threshold (or value)
    :param impurity: impurity
    :param n_node_samples: number of samples this node represents
    :param weighted_n_node_samples: node weight
    """
    return _tree_add_node(tree, parent, is_left, is_leaf, feature, threshold,
                          impurity, n_node_samples, weighted_n_node_samples)
