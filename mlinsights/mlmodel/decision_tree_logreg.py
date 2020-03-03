"""
@file
@brief Builds a tree of logistic regressions.
"""
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model._base import LinearClassifierMixin


def logistic(x):
    """
    Computes :math:`\\frac{1}{1 + e^{-x}}`.
    """
    return 1. / (1. + numpy.exp(-x))


def likelihood(x, y, theta=1., th=0.):
    """
    Computes :math:`\\sum_i y_i f(\\theta (x_i - x_0)) + (1 - y_i) (1 - f(\\theta (x_i - x_0)))`
    where :math:`f(x_i)` is :math:`\\frac{1}{1 + e^{-x}}`.
    """
    lr = logistic((x - th) * theta)
    return y * lr + (1. - y) * (1 - lr)


class _DecisionTreeLogisticRegressionNode:
    """
    Describes the tree structure hold by class
    @see cl DecisionTreeLogisticRegression.
    See also notebook :ref:`decisiontreelogregrst`.
    """

    def __init__(self, estimator, threshold=0.5, depth=1):
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

    def fit(self, X, y, sample_weight, dtlr, total_N):
        """
        Fits a logistic regression, then splits the sample into
        positive and negative examples, finally tries to fit
        logistic regressions on both subsamples. This method only
        works on a linear classifier.

        @param      X               features
        @param      y               binary labels
        @param      sample_weight   weights of every sample
        @param      dtlr            @see cl DecisionTreeLogisticRegression
        @param      total_N         total number of observation
        """
        self.estimator.fit(X, y, sample_weight=sample_weight)
        prob = self.fit_improve(dtlr, total_N, X, y,
                                sample_weight=sample_weight)

        if self.depth + 1 >= dtlr.max_depth:
            return
        if X.shape[0] < dtlr.min_samples_split:
            return

        above = prob[:, 1] > self.threshold
        below = ~ above
        n_above = above.sum()
        n_below = below.sum()
        y_above = set(y[above])
        y_below = set(y[below])

        def _fit_side(y_above_below, above_below, n_above_below):
            if (len(y_above_below) > 1 and
                    above_below.shape[0] > dtlr.min_samples_leaf * 2 and
                    (float(n_above_below) / total_N >=
                        dtlr.min_weight_fraction_leaf * 2) and
                    n_above_below < total_N):
                estimator = clone(dtlr.estimator)
                sw = sample_weight[above_below] if sample_weight is not None else None
                node = _DecisionTreeLogisticRegressionNode(
                    estimator, self.threshold, depth=self.depth + 1)
                node.fit(X[above_below], y[above_below], sw, dtlr, total_N)
                return node
            return None

        self.above = _fit_side(y_above, above, n_above)
        self.below = _fit_side(y_below, below, n_below)

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

    def fit_improve(self, dtlr, total_N, X, y, sample_weight):
        """
        The method only works on a linear classifier, it changes
        the intercept in order to be within the constraints
        imposed by the *min_samples_leaf* and *min_weight_fraction_leaf*.
        The algorithm has a significant cost as it sorts every observation
        and chooses the best intercept.

        @param      dtlr            @see cl DecisionTreeLogisticRegression
        @param      total_N         total number of observations
        @param      X               features
        @param      y               labels
        @param      sample_weight   sample weight
        @return                     probabilities
        """
        if self.estimator is None:
            raise RuntimeError("Estimator was not trained.")
        prob = self.estimator.predict_proba(X)
        if dtlr.fit_improve_algo in (None, 'none'):
            return prob

        if not isinstance(self.estimator, LinearClassifierMixin):
            # The classifier is not linear and cannot be improved.
            if self.fit_improve_algo == 'intercept_sort_always':
                raise RuntimeError(
                    "The model is not linear ({}), "
                    "intercept cannot be improved.".format(self.estimator.__class__.__name__))
            return prob

        above = prob[:, 1] > self.threshold
        below = ~ above
        n_above = above.sum()
        n_below = below.sum()
        n_min = min(n_above, n_below)
        if ((n_min >= dtlr.min_samples_leaf or
                float(n_min) / total_N >= dtlr.min_weight_fraction_leaf) and
                dtlr.fit_improve_algo != 'intercept_sort_always'):
            return prob

        coef = self.estimator.coef_
        intercept = self.estimator.intercept_
        decision_function = (X @ coef.T).ravel()
        order = numpy.argsort(decision_function, axis=0)
        begin = dtlr.min_samples_leaf

        sorted_df = decision_function[order]
        sorted_y = decision_function[order]
        N = sorted_y.shape[0]

        best = None
        besti = None
        beta_best = None
        for i in range(begin, N - begin):
            beta = - sorted_df[i]
            like = numpy.sum(likelihood(decision_function + beta, y)) / N
            w = float(i * (N - i)) / N**2
            like += w * dtlr.gamma
            if besti is None or like > best:
                best = like
                besti = i
                beta_best = beta

        if beta_best is not None:
            self.estimator.intercept_ = beta_best
            prob = self.estimator.predict_proba(X)
        return prob


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

    fit_improve_algo: string, one of the following value:
        - `'auto'`: chooses the best option below, '`none'` for
          every non linear model, `'intercept_sort'` for linear models
        - '`none'`: does not nothing once the binary classifier is fit
        - `'intercept_sort'`: if one side of the classifier is too small,
          the method changes the best intercept possible verifying
          the constraints
        - `'intercept_sort_always'`: always chooses the best intercept
          possible

    gamma: weight before the coefficient :math:`p (1-p)`.
        When the model tries to improve the linear classifier,
        it looks a better intercept which maximizes the
        likelihood and verifies the constraints.
        In order to force the classifier to choose a value
        which splits the dataset into 2 almost equal folds,
        the function maximimes :math:`likelihood + \\gamma p (1 - p)`
        where *p* is the proportion of samples falling in the first
        fold.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    tree_ : Tree
        The underlying Tree object.
    """

    _fit_improve_algo_values = (
        None, 'none', 'auto', 'intercept_sort', 'intercept_sort_always')

    def __init__(self, estimator=None,
                 max_depth=20, min_samples_split=2,
                 min_samples_leaf=2, min_weight_fraction_leaf=0.0,
                 fit_improve_algo='auto', gamma=1.):
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
        self.fit_improve_algo = fit_improve_algo
        self.gamma = gamma

        if self.fit_improve_algo not in DecisionTreeLogisticRegression._fit_improve_algo_values:
            raise ValueError(
                "fit_improve_algo='{}' not in {}".format(
                    self.fit_improve_algo, DecisionTreeLogisticRegression._fit_improve_algo_values))

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
        if (sample_weight is not None and
                not isinstance(sample_weight, numpy.ndarray)):
            raise TypeError("'sample_weight' must be an array.")
        self.classes_ = numpy.array(sorted(set(y)))
        if len(self.classes_) != 2:
            raise RuntimeError(
                "The model only supports binary classification but labels are "
                "{}.".format(self.classes_))
        cls = (y == self.classes_[1]).astype(numpy.int32)
        estimator = clone(self.estimator)
        self.tree_ = _DecisionTreeLogisticRegressionNode(estimator, 0.5)
        self.tree_.fit(X, cls, sample_weight, self, X.shape[0])
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
