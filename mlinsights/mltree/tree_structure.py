"""
@file
@brief Helpers to investigate a tree structure.
"""
import numpy
from sklearn.tree._tree import TREE_LEAF  # pylint: disable=E0611


def _get_tree(obj):
    """
    Returns the tree object.
    """
    if hasattr(obj, "children_left"):
        return obj
    if hasattr(obj, "tree_"):
        return obj.tree_
    raise AttributeError(  # pragma: no cover
        "obj is no tree: {}".format(type(obj)))


def tree_leave_index(model):
    """
    Returns the indices of every leave in a tree.

    @param      model       something which has a member ``tree_``
    @return                 leave indices
    """
    tree = _get_tree(model)
    res = []
    for i in range(tree.node_count):
        if tree.children_left[i] == TREE_LEAF:
            res.append(i)
    return res


def tree_find_path_to_root(tree, i, parents=None):
    """
    Lists nodes involved into the path to find node *i*.

    @param      tree        tree
    @param      i           node index (``tree.nodes[i]``)
    @param      parents     precomputed parents (None -> calls @see fn tree_node_range)
    @return                 one array of size *(D, 2)* where *D* is the number of dimensions
    """
    tree = _get_tree(tree)
    path_i = [i]
    current_i = i
    while current_i in parents:
        current_i = parents[current_i]
        if current_i < 0:
            current_i = - current_i
        path_i.append(current_i)
    return list(reversed(path_i))


def tree_find_common_node(tree, i, j, parents=None):
    """
    Finds the common node to nodes *i* and *j*.

    @param      tree        tree
    @param      i           node index (``tree.nodes[i]``)
    @param      j           node index (``tree.nodes[j]``)
    @param      parents     precomputed parents (None -> calls @see fn tree_node_range)
    @return                 common root, remaining path to *i*, remaining path to *j*
    """
    tree = _get_tree(tree)
    if parents is None:
        parents = tree_node_parents(tree)
    path_i = tree_find_path_to_root(tree, i, parents)
    path_j = tree_find_path_to_root(tree, j, parents)
    for pos, (a, b) in enumerate(zip(path_i, path_j)):
        if a != b:
            return a, path_i[pos:], path_j[pos:]
    pi = parents.get(i, None)
    pj = parents.get(j, None)
    pos = min(len(path_i), len(path_j))
    if pi is not None and pi == j:
        return j, path_i[pos:], path_j[pos:]
    if pj is not None and pj == i:
        return i, path_i[pos:], path_j[pos:]
    raise RuntimeError(  # pragma: no cover
        "Paths are equal, i={} and j={} must be differet.".format(i, j))


def tree_node_parents(tree):
    """
    Returns a dictionary ``{node_id: parent_id}``.

    @param      tree        tree
    @return                 parents
    """
    tree = _get_tree(tree)
    parents = {}
    for i in range(tree.node_count):
        if tree.children_left[i] == TREE_LEAF:
            continue
        parents[tree.children_left[i]] = i
        parents[tree.children_right[i]] = -i
    return parents


def tree_node_range(tree, i, parents=None):
    """
    Determines the ranges for a node all dimensions.
    ``nan`` means infinity.

    @param      tree        tree
    @param      i           node index (``tree.nodes[i]``)
    @param      parents     precomputed parents (None -> calls @see fn tree_node_range)
    @return                 one array of size *(D, 2)* where *D* is the number of dimensions

    The following example shows what the function returns
    in case of simple grid in two dimensions.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.tree import DecisionTreeClassifier
        from mlinsights.mltree import tree_leave_index, tree_node_range

        X = numpy.array([[0, 0], [0, 1], [0, 2],
                         [1, 0], [1, 1], [1, 2],
                         [2, 0], [2, 1], [2, 2]])
        y = list(range(X.shape[0]))
        clr = DecisionTreeClassifier(max_depth=4)
        clr.fit(X, y)

        leaves = tree_leave_index(clr)
        ra = tree_node_range(clr, leaves[0])

        print(ra)
    """
    tree = _get_tree(tree)
    if parents is None:
        parents = tree_node_parents(tree)
    path = tree_find_path_to_root(tree, i, parents)
    mx = max([tree.feature[p] for p in path])
    res = numpy.full((mx + 1, 2), numpy.nan)
    for ind, p in enumerate(path):
        if p == i:
            break
        fn = tree.feature[p]
        lr = tree.children_left[p] == path[ind + 1]
        th = tree.threshold[p]
        if lr:
            res[fn, 1] = min(res[fn, 1], th) if not numpy.isnan(
                res[fn, 1]) else th
        else:
            res[fn, 0] = max(res[fn, 0], th) if not numpy.isnan(
                res[fn, 0]) else th
    return res


def predict_leaves(model, X):
    """
    Returns the leave every observations of *X*
    falls into.

    @param      model       a decision tree
    @param      X           observations
    @return                 array of leaves
    """
    if hasattr(model, 'get_leaves_index'):
        leaves_index = model.get_leaves_index()
    else:
        leaves_index = [i for i in range(len(model.tree_.children_left))
                        if model.tree_.children_left[i] == TREE_LEAF]
    leaves = model.decision_path(X)
    leaves = leaves[:, leaves_index]
    mat = numpy.argmax(leaves, 1)
    res = numpy.asarray(mat).ravel()
    res = numpy.array([leaves_index[r] for r in res])
    return res


def tree_leave_neighbors(model):
    """
    The function determines which leaves are neighbors.
    The method uses some memory as it creates creates a
    grid of the feature spaces, each split multiplies the
    number of cells by two.

    @param      model       a :epkg:`sklearn:tree:DecisionTreeRegressor`,
                            a :epkg:`sklearn:tree:DecisionTreeClassifier`,
                            a model which has a member ``tree_``
    @return                 a dictionary ``{(i, j): (dimension, x1, x2)}``,
                            *i, j* are node indices, if :math:`X_d * sign < th  * sign`,
                            the observations goes to node *i*, *j* otherwise,
                            *i < j*. The border is somewhere in the segment ``[x1, x2]``.

    The following example shows what the function returns
    in case of simple grid in two dimensions.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.tree import DecisionTreeClassifier
        from mlinsights.mltree import tree_leave_neighbors

        X = numpy.array([[0, 0], [0, 1], [0, 2],
                         [1, 0], [1, 1], [1, 2],
                         [2, 0], [2, 1], [2, 2]])
        y = list(range(X.shape[0]))
        clr = DecisionTreeClassifier(max_depth=4)
        clr.fit(X, y)

        nei = tree_leave_neighbors(clr)

        import pprint
        pprint.pprint(nei)
    """
    tree = _get_tree(model)

    # creates the coordinates of the grid

    features = {}
    for i in range(tree.node_count):
        fe = tree.feature[i]
        if fe < 0:
            # leave
            continue
        th = tree.threshold[i]
        if fe not in features:
            features[fe] = []
        features[fe].append(th)
    for fe in features:
        features[fe] = list(sorted(set(features[fe])))
    for fe, v in features.items():
        if len(v) == 1:
            d = abs(v[0]) / 10
            if d == v[0]:
                d = 1
            v.insert(0, v[0] - d)
            v.append(v[-1] + d)
        else:
            diff = [v[i + 1] - v[i] for i in range(len(v) - 1)]
            mdiff = min(diff)
            v.append(v[-1] + mdiff)
            v.insert(0, v[0] - mdiff)

    # predictions

    keys = list(sorted(features))
    pos = [0 for k in keys]
    shape = [len(features[k]) - 1 for k in keys]
    cells = numpy.full(shape, 0, numpy.int32)
    while pos[0] < len(features[keys[0]]) - 1:
        # evaluate
        xy = numpy.zeros((1, model.n_features_))
        for p, k in zip(pos, keys):
            xy[0, k] = (features[k][p] + features[k][p + 1]) / 2
        leave = predict_leaves(model, xy)
        cells[tuple(pos)] = leave[0]

        # next
        ind = len(pos) - 1
        pos[ind] += 1
        while ind > 0 and pos[ind] >= len(features[keys[ind]]) - 1:
            pos[ind] = 0
            ind -= 1
            pos[ind] += 1

    # neighbors

    neighbors = {}
    pos = [0 for k in keys]
    while pos[0] <= len(features[keys[0]]) - 1:
        # neighbors
        try:
            cl = cells[tuple(pos)]
        except IndexError:
            # outside the cube
            cl = None
        if cl is not None:
            for k in range(len(pos)):  # pylint: disable=C0200
                pos[k] += 1
                try:
                    cl2 = cells[tuple(pos)]
                except IndexError:
                    # outside the cube
                    pos[k] -= 1
                    continue
                if cl != cl2:
                    edge = (cl, cl2) if cl < cl2 else (cl2, cl)
                    if edge not in neighbors:
                        neighbors[edge] = []
                    xy = numpy.zeros((model.n_features_))
                    for p, f in zip(pos, keys):
                        xy[f] = (features[f][p] + features[f][p + 1]) / 2
                    x2 = tuple(xy)
                    pos[k] -= 1
                    p = pos[k]
                    key = keys[k]
                    xy[key] = (features[key][p] + features[key][p + 1]) / 2
                    x1 = tuple(xy)
                    neighbors[edge].append((key, x1, x2))
                else:
                    pos[k] -= 1

        # next

        ind = len(pos) - 1
        pos[ind] += 1
        while ind > 0 and pos[ind] >= len(features[keys[ind]]) - 1:
            pos[ind] = 0
            ind -= 1
            pos[ind] += 1

    return neighbors
