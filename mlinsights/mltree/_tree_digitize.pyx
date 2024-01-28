cimport numpy as cnp

cnp.import_array()

# ctypedef cnp.npy_intp intp_t
ctypedef double float64_t
from sklearn.utils._typedefs cimport intp_t

from sklearn.tree._tree cimport Tree

TREE_LEAF = -1
TREE_UNDEFINED = -2


cdef intp_t _tree_add_node(Tree tree,
                           intp_t parent,
                           bint is_left,
                           bint is_leaf,
                           intp_t feature,
                           float64_t threshold,
                           float64_t impurity,
                           intp_t n_node_samples,
                           float64_t weighted_n_node_samples,
                           char missing_go_to_left):
    if parent == -1:
        parent = TREE_UNDEFINED
    return tree._add_node(parent, is_left, is_leaf, feature,
                          threshold, impurity,
                          n_node_samples, weighted_n_node_samples,
                          missing_go_to_left)


def tree_add_node(tree, parent, is_left, is_leaf, feature, threshold,
                  impurity, n_node_samples, weighted_n_node_samples,
                  missing_go_to_left):
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
    :param missing_go_to_left: whether features have missing values
    """
    return _tree_add_node(tree, parent, is_left, is_leaf, feature, threshold,
                          impurity, n_node_samples, weighted_n_node_samples,
                          missing_go_to_left)
