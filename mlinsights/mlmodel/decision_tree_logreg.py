"""
@file
@brief Builds a tree of logistic regressions.
"""
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class _DecisionTreeLogisticRegressionNode:
    """
    Describes the tree structure hold by class
    @see cl DecisionTreeLogisticRegression.
    """

    def __init__(self, estimator, threshold=0.5, depth=0):
        """
        constructor

        @param      estimator       binary estimator
        """
        self.estimator = estimator
        self.above = None
        self.below = None
        self.threshold = threshold
        self.depth = depth

    def predict(self, X):
        """
        Predicts
        """
        prob = self.predict_proba(X)
        return (prob[:, 1] >= 0.5).astype(numpy.int32)

    def predict_proba(self, X):
        """
        Returns the classification probabilities.

        @param          X   features
        @return             probabilties
        """
        prob = self.estimator.predict_proba(X)
        above = prob[:, 1] > self.threshold
        below = ~ above
        if self.above is not None:
            prob_above = self.above.predict_proba(X[above])
            prob[above] = prob_above
        if self.below is not None:
            prob_below = self.below.predict_proba(X[below])
            prob[below] = prob_below
        return prob

    def fit(self, X, y, sample_weight, dtlr):
        """
        Fits every example followin
        """
        if self.depth >= dtlr.max_depth:
            return
        if X.shape[0] < dtlr.min_samples_split:
            return

        prob = self.estimator.predict_proba(X)
        above = prob[:, 1] > self.threshold
        below = ~ above
        n_above = above.sum()
        n_below = below.sum()
        y_above = set(y[above])
        y_below = set(y[below])

        if (len(y_above) > 1 and above.shape[0] > dtlr.min_samples_leaf and
                float(n_above) / X.shape[0] >= dtlr.min_weight_fraction_leaf and
                n_above < X.shape[0]):
            estimator = clone(dtlr.estimator)
            sw = sample_weight[above] if sample_weight is not None else None
            estimator.fit(X[above], y[above], sample_weight=sw)
            self.above = _DecisionTreeLogisticRegressionNode(
                estimator, self.threshold, depth=self.depth + 1)
            self.above.fit(X[above], y[above], sw, dtlr)

        if (len(y_below) > 1 and below.shape[0] > dtlr.min_samples_leaf and
                float(n_below) / X.shape[0] >= dtlr.min_weight_fraction_leaf and
                n_below < X.shape[0]):
            estimator = clone(dtlr.estimator)
            sw = sample_weight[below] if sample_weight is not None else None
            estimator.fit(X[below], y[below], sample_weight=sw)
            self.below = _DecisionTreeLogisticRegressionNode(
                estimator, self.threshold, depth=self.depth + 1)
            self.below.fit(X[below], y[below], sw, dtlr)

    @property
    def tree_depth_(self):
        """
        Returns the maximum depth of the tree.
        """
        dt = self.depth
        if self.above is not None:
            dt = max(dt, self.above.tree_depth_)
        if self.below is not None:
            dt = max(dt, self.below.tree_depth_)
        return dt


class DecisionTreeLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Fits a logistic regression, then fits two other
    logistic regression for every observation on both sides
    of the border. It goes one until a tree is built.
    It only handles a binary classification.
    The built tree cannot be deeper than the maximum recursion.

    Parameters
    ----------
    estimator: binary classification estimator,
        if empty, use a logistic regression, the theoritical
        model defined with a logistic regression but it could
        any binary classifier

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples. It must be below the maximum
        allowed recursion by python.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    tree_ : Tree
        The underlying Tree object.
    """

    def __init__(self, estimator=None,
                 # tree
                 max_depth=10, min_samples_split=10,
                 min_samples_leaf=10, min_weight_fraction_leaf=0.0):
        "constructor"
        ClassifierMixin.__init__(self)
        BaseEstimator.__init__(self)
        # logistic regression
        if estimator is None:
            self.estimator = LogisticRegression()
        else:
            self.estimator = estimator
        if max_depth is None:
            raise ValueError("'max_depth' cannot be None.")
        if max_depth > 1024:
            raise ValueError("'max_depth' must be <= 1024.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf

    def fit(self, X, y, sample_weight=None):
        """
        Builds the tree model.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample

        Returns
        -------
        self : returns an instance of self.

        Attributes
        ----------

        classes_: classes

        tree_: see @see cl _DecisionTreeLogisticRegressionNode
        """
        if not isinstance(X, numpy.ndarray):
            if hasattr(X, 'values'):
                X = X.values
        if not isinstance(X, numpy.ndarray):
            raise TypeError("'X' must be an array.")
        self.classes_ = numpy.array(sorted(set(y)))
        if len(self.classes_) != 2:
            raise RuntimeError(
                "The model only supports binary classification but labels are "
                "{}.".format(self.classes_))
        cls = (y == self.classes_[1]).astype(numpy.int32)
        estimator = clone(self.estimator)
        estimator.fit(X, cls, sample_weight=sample_weight)
        self.tree_ = _DecisionTreeLogisticRegressionNode(estimator, 0.5)
        self.tree_.fit(X, cls, sample_weight, self)
        return self

    def predict(self, X):
        """
        Runs the predictions.
        """
        labels = self.tree_.predict(X)
        return numpy.take(self.classes_, labels)

    def predict_proba(self, X):
        """
        Converts predictions into probabilities.
        """
        return self.tree_.predict_proba(X)

    def decision_function(self, X):
        """
        Calls *decision_function*.
        """
        raise NotImplementedError(
            "Decision function is not available for this model.")

    @property
    def tree_depth_(self):
        """
        Returns the maximum depth of the tree.
        """
        return self.tree_.tree_depth_ + 1
