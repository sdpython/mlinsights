"""
@file
@brief Implements a piecewise linear regression.
"""
import numpy
import numpy.random
import pandas
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    pass


def _fit_piecewise_estimator(i, model, X, y, sample_weight, association, nb_classes, random_state):
    ind = association == i
    if not numpy.any(ind):
        # No training example for this bucket.
        return None
    Xi = X[ind, :]
    yi = y[ind]
    sw = sample_weight[ind] if sample_weight is not None else None

    if nb_classes is not None and len(set(yi)) != nb_classes:
        # Issues a classifiers requires to have at least one example
        # of each class.
        if random_state is None:
            random_state = numpy.random.RandomState()  # pylint: disable=E1101
        addition = numpy.arange(len(ind))
        random_state.shuffle(addition)
        found = set(yi)
        allcl = set(y)
        res = []
        while len(found) < len(allcl):
            for ki in addition:
                if y[ki] not in found:
                    res.append(ki)
                    found.add(y[ki])
        ind = ind.copy()
        for ki in res:
            ind[ki] = True

        Xi = X[ind, :]
        yi = y[ind]
        sw = sample_weight[ind] if sample_weight is not None else None

    return model.fit(Xi, yi, sample_weight=sw)


def _predict_piecewise_estimator(i, est, X, association):
    ind = association == i
    if not numpy.any(ind):
        return None, None
    return ind, est.predict(X[ind, :])


def _predict_proba_piecewise_estimator(i, est, X, association):
    ind = association == i
    if not numpy.any(ind):
        return None, None
    return ind, est.predict_proba(X[ind, :])


def _decision_function_piecewise_estimator(i, est, X, association):
    ind = association == i
    if not numpy.any(ind):
        return None, None
    return ind, est.decision_function(X[ind, :])


class PiecewiseEstimator(BaseEstimator):
    """
    Uses a :epkg:`decision tree` to split the space of features
    into buckets and trains a linear regression on each of them.
    The second estimator can be a :epkg:`sklearn:linear_model:LinearRegression`
    for a regression or :epkg:`sklearn:linear_model:LogisticRegression`
    for a classifier. It can also be :epkg:`sklearn:dummy:DummyRegressor`
    :epkg:`sklearn:dummy:DummyClassifier` to just get the average on each bucket.
    When the buckets are defined by a decision tree and the
    estimator is linear, @see cl PiecewiseTreeRegressor optimizes
    the buckets based on the results of a linear regression.
    The accuracy is usually better.
    """

    def __init__(self, binner=None, estimator=None, n_jobs=None, verbose=False):
        """
        @param      binner              transformer or predictor which creates the buckets
        @param      estimator           predictor trained on every bucket
        @param      n_jobs              number of parallel jobs (for training and predicting)
        @param      verbose             boolean or use ``'tqdm'`` to use :epkg:`tqdm`
                                        to fit the estimators

        *binner* must be filled or must be:

        - ``'bins'``: the model :epkg:`sklearn:preprocessing:KBinsDiscretizer`
        - any instanciated model

        *estimator* allows the following values:

        - ``None``: the model is :epkg:`sklearn:linear_model:LinearRegression`
        - any instanciated model
        """
        BaseEstimator.__init__(self)
        if estimator is None:
            raise ValueError(  # pragma: no cover
                "estimator cannot be null.")
        if binner is None:
            raise TypeError(  # pragma: no cover
                "Unsupported options for binner=='tree' and model {}.".format(
                    type(estimator)))
        elif binner == "bins":
            binner = KBinsDiscretizer()
        self.binner = binner
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    @property
    def n_estimators_(self):
        """
        Returns the number of estimators = the number of buckets
        the data was split in.
        """
        return len(self.estimators_)

    def _mapping_train(self, X, binner):
        if hasattr(binner, "tree_"):
            tree = binner.tree_
            leaves = [i for i in range(len(tree.children_left))
                      if tree.children_left[i] <= i and tree.children_right[i] <= i]
            dec_path = self.binner_.decision_path(X)
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

        elif hasattr(binner, "transform"):
            tr = binner.transform(X)
            unique = set()
            for x in tr:
                d = tuple(numpy.asarray(
                    x.todense()).ravel().astype(numpy.int32))
                unique.add(d)
            leaves = list(sorted(unique))
            association = numpy.zeros((X.shape[0],))
            association[:] = -1
            ntree = 0
            mapping = {}
            for i, le in enumerate(leaves):
                mapping[le] = i
            for i, x in enumerate(tr):
                d = tuple(numpy.asarray(
                    x.todense()).ravel().astype(numpy.int32))
                association[i] = mapping.get(d, -1)
        else:
            raise NotImplementedError(  # pragma: no cover
                "binner is not a decision tree or a transform")

        return association, mapping, leaves

    def transform_bins(self, X):
        """
        Maps every row to a tree in *self.estimators_*.
        """
        binner = self.binner_
        if hasattr(binner, "tree_"):
            dec_path = self.binner_.decision_path(X)
            association = numpy.zeros((X.shape[0],))
            association[:] = -1
            for j in self.leaves_:
                ind = dec_path[:, j] == 1
                ind = numpy.asarray(ind.todense()).flatten()
                if not numpy.any(ind):
                    # No training example for this bucket.
                    continue
                association[ind] = self.mapping_.get(j, -1)

        elif hasattr(binner, "transform"):
            association = numpy.zeros((X.shape[0],))
            association[:] = -1
            tr = binner.transform(X)
            for i, x in enumerate(tr):
                d = tuple(numpy.asarray(
                    x.todense()).ravel().astype(numpy.int32))
                association[i] = self.mapping_.get(d, -1)
        else:
            raise NotImplementedError(  # pragma: no cover
                "binner is not a decision tree or a transform")
        return association

    def fit(self, X, y, sample_weight=None):
        """
        Trains the binner and an estimator on every
        bucket.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :param y: target
        :param sample_weight: sample weights
        :return: self: returns an instance of self.

        Fitted attributes:

        * `binner_`: binner
        * `estimators_`: dictionary of estimators, each of them
            mapped to a leave to the tree
        * `mean_estimator_`: estimator trained on the whole
            datasets in case the binner can find a bucket for
            a new observation
        * `dim_`: dimension of the output
        * `mean_`: average targets
        """
        if len(y.shape) == 2:
            if y.shape[-1] == 1:
                y = y.ravel()
            else:
                raise RuntimeError(
                    "This regressor only works with single dimension targets.")
        if isinstance(X, pandas.DataFrame):
            X = X.values
        if isinstance(X, list):
            raise TypeError(  # pragma: no cover
                "X cannot be a list.")
        binner = clone(self.binner)
        if sample_weight is None:
            self.binner_ = binner.fit(X, y)
        else:
            self.binner_ = binner.fit(X, y, sample_weight=sample_weight)

        association, self.mapping_, self.leaves_ = self._mapping_train(
            X, self.binner_)

        estimators = [clone(self.estimator) for i in self.mapping_]

        loop = (tqdm(range(len(estimators)))
                if self.verbose == 'tqdm' else range(len(estimators)))
        verbose = 1 if self.verbose == 'tqdm' else (1 if self.verbose else 0)

        self.mean_estimator_ = clone(self.estimator).fit(X, y, sample_weight)
        nb_classes = (None if not hasattr(self.mean_estimator_, 'classes_')
                      else len(set(self.mean_estimator_.classes_)))

        if hasattr(self, 'random_state') and self.random_state is not None:  # pylint: disable=E1101
            rnd = numpy.random.RandomState(  # pylint: disable=E1101
                self.random_state)  # pylint: disable=E1101
        else:
            rnd = None

        self.estimators_ = \
            Parallel(n_jobs=self.n_jobs, verbose=verbose,
                     **_joblib_parallel_args(prefer='threads'))(
                delayed(_fit_piecewise_estimator)(
                    i, estimators[i], X, y, sample_weight, association, nb_classes, rnd)
                for i in loop)

        self.dim_ = 1 if len(y.shape) == 1 else y.shape[1]
        if hasattr(self.estimators_[0], 'classes_'):
            self.classes_ = self.estimators_[0].classes_
        return self

    def _apply_predict_method(self, X, method, parallelized, dimout):
        """
        Generic *predict* method, works for *predict_proba* and
        *decision_function* as well.
        """
        if len(self.estimators_) == 0:
            raise RuntimeError(  # pragma: no cover
                "Estimator was apparently fitted but contains no estimator.")
        if not hasattr(self.estimators_[0], method):
            raise TypeError(  # pragma: no cover
                "Estimator {} does not have method '{}'.".format(
                    type(self.estimators_[0]), method))
        if isinstance(X, pandas.DataFrame):
            X = X.values

        association = self.transform_bins(X)

        indpred = Parallel(n_jobs=self.n_jobs, **_joblib_parallel_args(prefer='threads'))(
            delayed(parallelized)(i, model, X, association)
            for i, model in enumerate(self.estimators_))

        pred = numpy.zeros((X.shape[0], dimout)
                           if dimout > 1 else (X.shape[0],))
        indall = numpy.empty((X.shape[0],))
        indall[:] = False
        for ind, p in indpred:
            if ind is None:
                continue
            pred[ind] = p
            indall = numpy.logical_or(indall, ind)  # pylint: disable=E1111

        # no in a bucket
        indall = numpy.logical_not(indall)  # pylint: disable=E1111
        Xmissed = X[indall]
        if Xmissed.shape[0] > 0:
            meth = getattr(self.mean_estimator_, method)
            missed = meth(Xmissed)
            pred[indall] = missed
        return pred


class PiecewiseRegressor(PiecewiseEstimator, RegressorMixin):
    """
    Uses a :epkg:`decision tree` to split the space of features
    into buckets and trains a linear regression (default) on each of them.
    The second estimator is usually a :epkg:`sklearn:linear_model:LinearRegression`.
    It can also be :epkg:`sklearn:dummy:DummyRegressor` to just get
    the average on each bucket.
    """

    def __init__(self, binner=None, estimator=None, n_jobs=None, verbose=False):
        """
        @param      binner              transformer or predictor which creates the buckets
        @param      estimator           predictor trained on every bucket
        @param      n_jobs              number of parallel jobs (for training and predicting)
        @param      verbose             boolean or use ``'tqdm'`` to use :epkg:`tqdm`
                                        to fit the estimators

        *binner* allows the following values:

        - ``tree``: the model is :epkg:`sklearn:tree:DecisionTreeRegressor`
        - ``'bins'``: the model :epkg:`sklearn:preprocessing:KBinsDiscretizer`
        - any instanciated model

        *estimator* allows the following values:

        - ``None``: the model is :epkg:`sklearn:linear_model:LinearRegression`
        - any instanciated model
        """
        if estimator is None:
            estimator = LinearRegression()
        if binner in ('tree', None):
            binner = DecisionTreeRegressor(min_samples_leaf=2)
        RegressorMixin.__init__(self)
        PiecewiseEstimator.__init__(self, binner=binner, estimator=estimator,
                                    n_jobs=n_jobs, verbose=verbose)

    def predict(self, X):
        """
        Computes the predictions.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :return: predictions
        """
        return self._apply_predict_method(
            X, "predict", _predict_piecewise_estimator, self.dim_)


class PiecewiseClassifier(PiecewiseEstimator, ClassifierMixin):
    """
    Uses a :epkg:`decision tree` to split the space of features
    into buckets and trains a logistic regression (default) on each of them.
    The second estimator is usually a :epkg:`sklearn:linear_model:LogisticRegression`.
    It can also be :epkg:`sklearn:dummy:DummyClassifier` to just get
    the average on each bucket.

    The main issue with the *PiecewiseClassifier* is that each piece requires
    one example of each class in each bucket which may not happen.
    To avoid that, the training will pick up random example
    from other bucket to ensure this case does not happen.
    """

    def __init__(self, binner=None, estimator=None, n_jobs=None,
                 random_state=None, verbose=False):
        """
        @param      binner              transformer or predictor which creates the buckets
        @param      estimator           predictor trained on every bucket
        @param      n_jobs              number of parallel jobs (for training and predicting)
        @param      random_state        to pick up random examples when buckets do not
                                        contain enough examples of each class
        @param      verbose             boolean or use ``'tqdm'`` to use :epkg:`tqdm`
                                        to fit the estimators

        *binner* allows the following values:

        - ``tree``: the model is :epkg:`sklearn:tree:DecisionTreeClassifier`
        - ``'bins'``: the model :epkg:`sklearn:preprocessing:KBinsDiscretizer`
        - any instanciated model

        *estimator* allows the following values:

        - ``None``: the model is :epkg:`sklearn:linear_model:LogisticRegression`
        - any instanciated model
        """
        if estimator is None:
            estimator = LogisticRegression()
        if binner in ('tree', None):
            binner = DecisionTreeClassifier(min_samples_leaf=5)
        ClassifierMixin.__init__(self)
        PiecewiseEstimator.__init__(
            self, binner=binner, estimator=estimator,
            n_jobs=n_jobs, verbose=verbose)
        self.random_state = random_state

    def predict(self, X):
        """
        Computes the predictions.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :return: predictions
        """
        pred = self._apply_predict_method(
            X, "predict", _predict_piecewise_estimator, 1)
        return pred.astype(numpy.int32)

    def predict_proba(self, X):
        """
        Computes the predictions probabilities.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :return: predictions probabilities
        """
        return self._apply_predict_method(
            X, "predict_proba", _predict_proba_piecewise_estimator,
            len(self.mean_estimator_.classes_))

    def decision_function(self, X):
        """
        Computes the predictions probabilities.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :return: predictions probabilities
        """
        justone = self.mean_estimator_.decision_function(X[:1])
        return self._apply_predict_method(
            X, "decision_function", _decision_function_piecewise_estimator,
            1 if len(justone.shape) == 1 else justone.shape[1])
