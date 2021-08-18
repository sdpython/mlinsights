"""
@file
@brief Builds a tree of logistic regressions.
"""
import numpy
import scipy.sparse as sparse  # pylint: disable=R0402
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

    def __init__(self, estimator, threshold=0.5, depth=1, index=0):
        """
        constructor

        @param      estimator       binary estimator
        """
        self.index = index
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
        n_above = above.sum()
        n_below = below.sum()
        if self.above is not None and n_above > 0:
            prob_above = self.above.predict_proba(X[above])
            prob[above] = prob_above
        if self.below is not None and n_below > 0:
            prob_below = self.below.predict_proba(X[below])
            prob[below] = prob_below
        return prob

    def decision_path(self, X, mat, indices):
        """
        Returns the classification probabilities.

        @param          X       features
        @param          mat     decision path (allocated matrix)
        """
        mat[indices, self.index] = 1
        prob = self.estimator.predict_proba(X)
        above = prob[:, 1] > self.threshold
        below = ~ above
        n_above = above.sum()
        n_below = below.sum()
        indices_above = indices[above]
        indices_below = indices[below]
        if self.above is not None and n_above > 0:
            self.above.decision_path(X[above], mat, indices_above)
        if self.below is not None and n_below > 0:
            self.below.decision_path(X[below], mat, indices_below)

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
        @return                     last index
        """
        self.estimator.fit(X, y, sample_weight=sample_weight)
        if dtlr.verbose >= 1:
            print("[DTLR ] %s trained acc %1.2f N=%d" % (  # pragma: no cover
                " " * self.depth, self.estimator.score(X, y), X.shape[0]))
        prob = self.fit_improve(dtlr, total_N, X, y,
                                sample_weight=sample_weight)

        if self.depth + 1 > dtlr.max_depth:
            return self.index
        if X.shape[0] < dtlr.min_samples_split:
            return self.index

        above = prob[:, 1] > self.threshold
        below = ~ above
        n_above = above.sum()
        n_below = below.sum()
        y_above = set(y[above])
        y_below = set(y[below])

        def _fit_side(index, y_above_below, above_below, n_above_below, side):
            if dtlr.verbose >= 1:
                print("[DTLR*] %s%s: n_class=%d N=%d - %d/%d" % (  # pragma: no cover
                    " " * self.depth, side,
                    len(y_above_below), above_below.shape[0],
                    n_above_below, total_N))
            if (len(y_above_below) > 1 and
                    above_below.shape[0] > dtlr.min_samples_leaf * 2 and
                    (float(n_above_below) / total_N >=
                        dtlr.min_weight_fraction_leaf * 2) and
                    n_above_below < total_N):
                estimator = clone(dtlr.estimator)
                sw = sample_weight[above_below] if sample_weight is not None else None
                node = _DecisionTreeLogisticRegressionNode(
                    estimator, self.threshold, depth=self.depth + 1, index=index)
                last_index = node.fit(
                    X[above_below], y[above_below], sw, dtlr, total_N)
                return node, last_index
            return None, index

        self.above, last = _fit_side(
            self.index + 1, y_above, above, n_above, "above")
        self.below, last = _fit_side(
            last + 1, y_below, below, n_below, "below")
        return last

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
            raise RuntimeError(
                "Estimator was not trained.")  # pragma: no cover
        prob = self.estimator.predict_proba(X)
        if dtlr.fit_improve_algo in (None, 'none'):
            return prob

        if not isinstance(self.estimator, LinearClassifierMixin):
            # The classifier is not linear and cannot be improved.
            if dtlr.fit_improve_algo == 'intercept_sort_always':  # pragma: no cover
                raise RuntimeError(
                    "The model is not linear ({}), "
                    "intercept cannot be improved.".format(self.estimator.__class__.__name__))
            return prob

        above = prob[:, 1] > self.threshold
        below = ~ above
        n_above = above.sum()
        n_below = below.sum()
        n_min = min(n_above, n_below)
        p1p2 = float(n_above * n_below) / X.shape[0] ** 2
        if dtlr.verbose >= 2:
            print("[DTLRI] %s imp %d <> %d, p1p2=%1.3f <> %1.3f" % (  # pragma: no cover
                " " * self.depth, n_min, dtlr.min_samples_leaf,
                p1p2, dtlr.p1p2))
        if (n_min >= dtlr.min_samples_leaf and
                float(n_min) / total_N >= dtlr.min_weight_fraction_leaf and
                p1p2 > dtlr.p1p2 and
                dtlr.fit_improve_algo != 'intercept_sort_always'):
            return prob

        coef = self.estimator.coef_
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
            if dtlr.verbose >= 1:
                print("[DTLRI] %s change intercept %f --> %f in [%f, %f]" % (  # pragma: no cover
                    " " * self.depth, self.estimator.intercept_, beta_best,
                    - sorted_df[-1], - sorted_df[0]))
            self.estimator.intercept_ = beta_best
            prob = self.estimator.predict_proba(X)
        return prob

    def enumerate_leaves_index(self):
        """
        Returns the leaves index.
        """
        if self.above is None or self.below is None:
            yield self.index
        if self.above is not None:
            for index in self.above.enumerate_leaves_index():
                yield index
        if self.below is not None:
            for index in self.below.enumerate_leaves_index():
                yield index


class DecisionTreeLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Fits a logistic regression, then fits two other
    logistic regression for every observation on both sides
    of the border. It goes one until a tree is built.
    It only handles a binary classification.
    The built tree cannot be deeper than the maximum recursion.

    :param estimator: binary classification estimator,
        if empty, use a logistic regression, the theoritical
        model defined with a logistic regression but it could
        any binary classifier
    :param max_depth: int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples. It must be below the maximum
        allowed recursion by python.
    :param min_samples_split: int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    :param min_samples_leaf: int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    :param min_weight_fraction_leaf: float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    :param fit_improve_algo: string, one of the following value:
        - `'auto'`: chooses the best option below, '`none'` for
          every non linear model, `'intercept_sort'` for linear models
        - '`none'`: does not nothing once the binary classifier is fit
        - `'intercept_sort'`: if one side of the classifier is too small,
          the method changes the best intercept possible verifying
          the constraints
        - `'intercept_sort_always'`: always chooses the best intercept
          possible
    :param p1p2: threshold in [0, 1]
        for every split, we can define probabilities :math:`p_1 p_2`
        which define the ratio of samples in both splits,
        if :math:`p_1 p_2` is below the threshold,
        method *fit_improve* is called
    :param gamma: weight before the coefficient :math:`p (1-p)`.
        When the model tries to improve the linear classifier,
        it looks a better intercept which maximizes the
        likelihood and verifies the constraints.
        In order to force the classifier to choose a value
        which splits the dataset into 2 almost equal folds,
        the function maximimes :math:`likelihood + \\gamma p (1 - p)`
        where *p* is the proportion of samples falling in the first
        fold.
    :param verbose: prints out information about the training
    :param strategy: `'parallel'` or `'perpendicular'`,
        see below

    Fitted attributes:

    *  `classes_`: ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).
    * `tree_`: Tree
        The underlying Tree object.

    The class implements two strategies to build the tree.
    The first one `'parallel'` splits the feature space using
    the hyperplan defined by a logistic regression, the second
    strategy `'perpendicular'` splis the feature space based on
    a hyperplan perpendicular to a logistic regression. By doing
    this, two logistic regression fit on both sub parts must
    necessary decreases the training error.
    """

    _fit_improve_algo_values = (
        None, 'none', 'auto', 'intercept_sort', 'intercept_sort_always')

    def __init__(self, estimator=None,
                 max_depth=20, min_samples_split=2,
                 min_samples_leaf=2, min_weight_fraction_leaf=0.0,
                 fit_improve_algo='auto', p1p2=0.09,
                 gamma=1., verbose=0, strategy='parallel'):
        "constructor"
        ClassifierMixin.__init__(self)
        BaseEstimator.__init__(self)
        # logistic regression
        if estimator is None:
            self.estimator = LogisticRegression()
        else:
            self.estimator = estimator
        if max_depth is None:
            raise ValueError("'max_depth' cannot be None.")  # pragma: no cover
        if max_depth > 1024:
            raise ValueError(
                "'max_depth' must be <= 1024.")  # pragma: no cover
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.fit_improve_algo = fit_improve_algo
        self.p1p2 = p1p2
        self.gamma = gamma
        self.verbose = verbose
        self.strategy = strategy

        if self.fit_improve_algo not in DecisionTreeLogisticRegression._fit_improve_algo_values:
            raise ValueError(
                "fit_improve_algo='{}' not in {}".format(
                    self.fit_improve_algo, DecisionTreeLogisticRegression._fit_improve_algo_values))

    def fit(self, X, y, sample_weight=None):
        """
        Builds the tree model.

        :param X: numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        :param y: numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary
        :param sample_weight: numpy array of shape [n_samples]
            Individual weights for each sample
        :return: self : returns an instance of self.

        Fitted attributes:

        * `classes_`: classes
        * `tree_`: tree structure, see @see cl _DecisionTreeLogisticRegressionNode
        * `n_nodes_`: number of nodes
        """
        if not isinstance(X, numpy.ndarray):
            if hasattr(X, 'values'):
                X = X.values
        if not isinstance(X, numpy.ndarray):
            raise TypeError("'X' must be an array.")
        if (sample_weight is not None and
                not isinstance(sample_weight, numpy.ndarray)):
            raise TypeError(
                "'sample_weight' must be an array.")  # pragma: no cover
        self.classes_ = numpy.array(sorted(set(y)))
        if len(self.classes_) != 2:
            raise RuntimeError(
                "The model only supports binary classification but labels are "
                "{}.".format(self.classes_))

        if self.strategy == 'parallel':
            return self._fit_parallel(X, y, sample_weight)
        if self.strategy == 'perpendicular':
            return self._fit_perpendicular(X, y, sample_weight)
        raise ValueError(
            "Unknown strategy '{}'.".format(self.strategy))

    def _fit_parallel(self, X, y, sample_weight):
        "Implements the parallel strategy."
        cls = (y == self.classes_[1]).astype(numpy.int32)
        estimator = clone(self.estimator)
        self.tree_ = _DecisionTreeLogisticRegressionNode(estimator, 0.5)
        self.n_nodes_ = self.tree_.fit(
            X, cls, sample_weight, self, X.shape[0]) + 1
        return self

    def _fit_perpendicular(self, X, y, sample_weight):
        "Implements the perpendicular strategy."
        raise NotImplementedError()  # pragma: no cover

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
        raise NotImplementedError(  # pragma: no cover
            "Decision function is not available for this model.")

    @property
    def tree_depth_(self):
        """
        Returns the maximum depth of the tree.
        """
        return self.tree_.tree_depth_

    def decision_path(self, X, check_input=True):
        """
        Returns the decision path.

        @param      X               inputs
        @param      check_input     unused
        @return                     sparse matrix
        """
        mat = sparse.lil_matrix((X.shape[0], self.n_nodes_), dtype=numpy.int32)
        self.tree_.decision_path(X, mat, numpy.arange(X.shape[0]))
        return sparse.csr_matrix(mat)

    def get_leaves_index(self):
        """
        Returns the index of every leave.
        """
        indices = self.tree_.enumerate_leaves_index()
        return numpy.array(sorted(indices))
