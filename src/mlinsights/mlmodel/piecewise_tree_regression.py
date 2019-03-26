# -*- coding: utf-8 -*-
"""
@file
@brief Implements a kind of piecewise linear regression by modifying
the criterion used by the algorithm which builds a decision tree.
"""
import sklearn
from sklearn.tree import DecisionTreeRegressor
from pyquickhelper.texthelper import compare_module_version


class DecisionTreeLinearRegressor(DecisionTreeRegressor):
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
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 presort=False):
        DecisionTreeRegressor.__init__(self, criterion=criterion,
                                       splitter=splitter, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                       random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                       min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                                       presort=presort)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        replace = None
        if isinstance(self.criterion, str):
            if self.criterion == 'mselin':
                if compare_module_version(sklearn.__version__, '0.21') >= 0:
                    from .piecewise_tree_regression_criterion_linear import LinearRegressorCriterion  # pylint: disable=E0611
                    self.criterion = LinearRegressorCriterion(X)
                    replace = self.criterion
                else:
                    raise ImportError(
                        "LinearRegressorCriterion only exists for scikit-learn >= 0.21.")
            elif self.criterion == "simple":
                if compare_module_version(sklearn.__version__, '0.21') >= 0:
                    from .piecewise_tree_regression_criterion_fast import SimpleRegressorCriterionFast  # pylint: disable=E0611
                    replace = self.criterion
                    self.criterion = SimpleRegressorCriterionFast(X)
                else:
                    raise ImportError(
                        "SimpleRegressorCriterion only exists for scikit-learn >= 0.21.")
        else:
            replace = None

        DecisionTreeRegressor.fit(self, X, y, sample_weight=sample_weight, check_input=check_input,
                                  X_idx_sorted=X_idx_sorted)

        if replace:
            self.criterion = replace
        return self
