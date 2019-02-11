"""
@file
@brief Implements a quantile linear regression.
"""
import numpy
import pandas
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted
try:
    from tqdm import tqdm
except ImportError:
    pass


class PiecewiseLinearRegression(BaseEstimator, RegressorMixin):
    """
    Uses a :epkg:`DeecisionTree` to split the space of features
    into buckets and trains a linear regression on each of them.
    The second estimator is usually a :epkg:`sklearn:linear_model:LinearRegression`.
    It can also be :epkg:`sklearn:dummy:DummyRegressor` to just get
    the average on each bucket.
    """

    def __init__(self, binner=None, estimator=None, verbose=False):
        """
        @param      binner              transformer or predictor which creates the buckets
        @param      estimator           predictor trained on every bucket
        @param      verbose             use :epkg:`tqdm` to fit the estimators

        *binner* allows the following values:
        * ``None``: the model is :epkg:`sklearn:tree:DecisionTreeRegressor`
        * ``'bins'``: the model :epkg:`sklearn:preprocessing:KBinsDiscretizer`
        * any instanciated model

        *estimator* allows the following values:
        * ``None``: the model is :epkg:`sklearn:linear_model:LinearRegression`
        * any instanciated model
        """
        RegressorMixin.__init__(self)
        BaseEstimator.__init__(self)
        if binner is None:
            binner = DecisionTreeRegressor(min_samples_leaf=2)
        elif binner == "bins":
            binner = KBinsDiscretizer()
        if estimator is None:
            estimator = LinearRegression()
        self.binner = binner
        self.estimator = estimator
        self.verbose = verbose

    @property
    def n_estimators_(self):
        check_is_fitted(self, 'estimators_')
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
            raise NotImplementedError(
                "binner is not a decision tree or a transform")

        return association, mapping, leaves

    def transform_bins(self, X):
        """
        Maps every row to a tree in *self.estimators_*.
        """
        check_is_fitted(self, 'mapping_')
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
            raise NotImplementedError(
                "binner is not a decision tree or a transform")
        return association

    def fit(self, X, y, sample_weight=None):
        """
        Trains the binner and an estimator on every
        bucket.

        Parameters
        ----------
        X: features, *X* is converted into an array if *X* is a dataframe

        y: target

        sample_weight: sample weights

        Returns
        -------
        self: returns an instance of self.

        Attributes
        ----------

        binner_: binner

        estimators_: dictionary of estimators, each of them
            mapped to a leave to the tree

        dim_: dimension of the output
        mean_: average targets
        """
        if isinstance(X, pandas.DataFrame):
            X = X.values
        if isinstance(X, list):
            raise TypeError("X cannot be a list.")
        binner = clone(self.binner)
        if sample_weight is None:
            self.binner_ = binner.fit(X, y)
        else:
            self.binner_ = binner.fit(X, y, sample_weight=sample_weight)
        self.estimators_ = {}
        association, self.mapping_, self.leaves_ = self._mapping_train(
            X, self.binner_)

        def fit_estimator(i):
            ind = association == i
            if not numpy.any(ind):
                # No training example for this bucket.
                return None
            Xi = X[ind, :]
            yi = y[ind]
            sw = sample_weight[ind] if sample_weight is not None else None
            model = clone(self.estimator).fit(Xi, yi, sample_weight=sw)
            return model

        self.estimators_ = [None for i in self.mapping_]

        if self.verbose:
            for i in tqdm(range(len(self.estimators_))):
                est = fit_estimator(i)
                self.estimators_[i] = est
        else:
            for i in range(len(self.estimators_)):
                est = fit_estimator(i)
                self.estimators_[i] = est
        self.dim_ = 1 if len(y.shape) == 1 else y.shape[1]
        self.mean_ = numpy.average(y, weights=sample_weight)
        return self

    def predict(self, X):
        """
        Runs the predictions.

        Parameters
        ----------
        X: features, *X* is converted into an array if *X* is a dataframe

        Returns
        -------

        predictions
        """
        check_is_fitted(self, 'estimators_')
        if isinstance(X, pandas.DataFrame):
            X = X.values

        association = self.transform_bins(X)

        pred = numpy.zeros((X.shape[0], self.dim_)
                           if self.dim_ > 1 else (X.shape[0],))
        pred[:] = self.mean_
        for ntree, est in enumerate(self.estimators_):
            ind = association == ntree
            if not numpy.any(ind):
                continue
            pred[ind] = est.predict(X[ind, :])
        return pred
