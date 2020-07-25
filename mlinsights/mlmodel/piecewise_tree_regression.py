# -*- coding: utf-8 -*-
"""
@file
@brief Implements a kind of piecewise linear regression by modifying
the criterion used by the algorithm which builds a decision tree.
"""
import numpy
from sklearn.tree import DecisionTreeRegressor


class PiecewiseTreeRegressor(DecisionTreeRegressor):
    """
    Implements a kind of piecewise linear regression by modifying
    the criterion used by the algorithm which builds a decision tree.
    See :epkg:`sklearn:tree:DecisionTreeRegressor` to get the meaning
    of the parameters except criterion:

    * ``mselin``: optimizes for a piecewise linear regression
    * ``simple``: optimizes for a stepwise regression (equivalent to *mse*)
    """

    def __init__(self, criterion='mselin', splitter='best', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None):
        DecisionTreeRegressor.__init__(
            self, criterion=criterion,
            splitter=splitter, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        replace = None
        if isinstance(self.criterion, str):
            if self.criterion == 'mselin':
                from .piecewise_tree_regression_criterion_linear import (  # pylint: disable=E0611,C0415
                    LinearRegressorCriterion)
                replace = self.criterion
                self.criterion = LinearRegressorCriterion(X)
            elif self.criterion == "simple":
                from .piecewise_tree_regression_criterion_fast import (  # pylint: disable=E0611,C0415
                    SimpleRegressorCriterionFast)
                replace = self.criterion
                self.criterion = SimpleRegressorCriterionFast(X)
        else:
            replace = None

        DecisionTreeRegressor.fit(self, X, y, sample_weight=sample_weight, check_input=check_input,
                                  X_idx_sorted=X_idx_sorted)

        if replace:
            self.criterion = replace

        if self.criterion == "mselin":
            self._fit_reglin(X, y, sample_weight)
        return self

    def _mapping_train(self, X):
        tree = self.tree_
        leaves = [i for i in range(len(tree.children_left))
                  if tree.children_left[i] <= i and tree.children_right[i] <= i]  # pylint: disable=E1136
        dec_path = self.decision_path(X)
        association = numpy.zeros((X.shape[0],))
        association[:] = -1
        mapping = {}
        ntree = 0
        for j in leaves:
            ind = dec_path[:, j] == 1
            ind = numpy.asarray(ind.todense()).flatten()
            if not numpy.any(ind):
                # No training example for this bucket.
                continue
            mapping[j] = ntree
            association[ind] = ntree
            ntree += 1
        return mapping

    def predict_leaves(self, X):
        """
        Returns the leave index for each observation of *X*.

        :param X: array
        :return: array
            leaves index in ``self.leaves_index_``
        """
        # The creation of the sparse matrix could be avoided.
        leaves = self.decision_path(X)
        leaves = leaves[:, self.leaves_index_]
        mat = numpy.argmax(leaves, 1)
        res = numpy.asarray(mat).ravel()
        return res

    def _fit_reglin(self, X, y, sample_weight):
        """
        Fits linear regressions for all leaves.
        Sets attributes ``leaves_mapping_``, ``betas_``, ``leaves_index_``.
        The first attribute is a dictionary ``{leave: row}``
        which maps a leave of the tree to the coefficients
        ``betas_[row, :]`` of a regression trained on all training
        points mapped a specific leave. ``leaves_index_`` keeps
        in memory a set of leaves.
        """
        from .piecewise_tree_regression_criterion_linear import (  # pylint: disable=E0611,C0415
            LinearRegressorCriterion)

        tree = self.tree_
        self.leaves_index_ = [i for i in range(len(tree.children_left))
                              if tree.children_left[i] <= i and tree.children_right[i] <= i]  # pylint: disable=E1136
        if tree.n_leaves != len(self.leaves_index_):
            raise RuntimeError(  # pragma: no cover
                "Unexpected number of leaves {} != {}".format(
                    tree.n_leaves, len(self.leaves_index_)))
        pred_leaves = self.predict_leaves(X)
        self.leaves_mapping_ = {k: i for i, k in enumerate(pred_leaves)}
        self.betas_ = numpy.empty((len(self.leaves_index_), X.shape[1] + 1))
        for i, _ in enumerate(self.leaves_index_):
            ind = pred_leaves == i
            xs = X[ind, :].copy()
            ys = y[ind].astype(numpy.float64)
            if len(ys.shape) == 1:
                ys = ys[:, numpy.newaxis]
            ys = ys.copy()
            ws = sample_weight[ind].copy() if sample_weight else None
            dec = LinearRegressorCriterion.create(xs, ys, ws)
            dec.node_beta(self.betas_[i, :])

    def predict(self, X, check_input=True):
        """
        Overloads method *predict*. Falls back into
        the predict from a decision tree is criterion is
        *mse*, *mae*, *simple*. Computes the predictions
        from linear regression if the criterion is *mselin*.
        """
        if self.criterion == 'mselin':
            return self._predict_reglin(X, check_input=check_input)
        return DecisionTreeRegressor.predict(self, X, check_input=check_input)

    def _predict_reglin(self, X, check_input=True):
        """
        Computes the predictions with a linear regression
        fitted with the observations mapped to each leave
        of the tree.

        :param X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        :param check_input: boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        :return: y, array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        leaves = self.predict_leaves(X)
        pred = numpy.ones((X.shape[0], 1))
        Xone = numpy.hstack([X, pred])
        for i in range(0, X.shape[0]):
            li = leaves[i]
            pred[i] = numpy.dot(Xone[i, :], self.betas_[li, :])
        return pred.ravel()
