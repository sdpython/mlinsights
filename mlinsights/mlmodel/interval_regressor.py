"""
@file
@brief Implements a piecewise linear regression.
"""
import numpy
import numpy.random
from sklearn.base import RegressorMixin, clone, BaseEstimator
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    pass


class IntervalRegressor(BaseEstimator, RegressorMixin):
    """
    Trains multiple regressors to provide a confidence
    interval on prediction. It only works for
    single regression. Every training is made with a new
    sample of the training data, parameter *alpha*
    let the user choose the size of this sample.
    A smaller *alpha* increases the variance
    of the predictions. The current implementation
    draws sample by random but keeps the weight associated
    to each of them. Another way could be to draw
    a weighted sample but give them uniform weights.
    """

    def __init__(self, estimator=None, n_estimators=10, n_jobs=None,
                 alpha=1., verbose=False):
        """
        @param      estimator           predictor trained on every bucket
        @param      n_estimators        number of estimators to train
        @param      n_jobs              number of parallel jobs (for training and predicting)
        @param      alpha               proportion of samples resampled for each training
        @param      verbose             boolean or use ``'tqdm'`` to use :epkg:`tqdm`
                                        to fit the estimators
        """
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)
        if estimator is None:
            raise ValueError("estimator cannot be null.")  # pragma: no cover
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.alpha = alpha
        self.verbose = verbose
        self.n_estimators = n_estimators

    @property
    def n_estimators_(self):
        """
        Returns the number of estimators = the number of buckets
        the data was split in.
        """
        return len(self.estimators_)

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
        self.estimators_ = []
        estimators = [clone(self.estimator) for i in range(self.n_estimators)]

        loop = tqdm(range(len(estimators))
                    ) if self.verbose == 'tqdm' else range(len(estimators))
        verbose = 1 if self.verbose == 'tqdm' else (1 if self.verbose else 0)

        def _fit_piecewise_estimator(i, est, X, y, sample_weight, alpha):
            new_size = int(X.shape[0] * alpha + 0.5)
            rnd = numpy.random.randint(0, X.shape[0] - 1, new_size)
            Xr = X[rnd]
            yr = y[rnd]
            sr = sample_weight[rnd] if sample_weight else None
            return est.fit(Xr, yr, sr)

        self.estimators_ = \
            Parallel(n_jobs=self.n_jobs, verbose=verbose,
                     **_joblib_parallel_args(prefer='threads'))(
                delayed(_fit_piecewise_estimator)(
                    i, estimators[i], X, y, sample_weight, self.alpha)
                for i in loop)

        return self

    def predict_all(self, X):
        """
        Computes the predictions for all estimators.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :return: predictions
        """
        container = numpy.empty((X.shape[0], len(self.estimators_)))
        for i, est in enumerate(self.estimators_):
            pred = est.predict(X)
            container[:, i] = pred
        return container

    def predict(self, X):
        """
        Computes the average predictions.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :return: predictions
        """
        preds = self.predict_all(X)
        return preds.mean(axis=1)

    def predict_sorted(self, X):
        """
        Computes the predictions for all estimators.
        Sorts them for all observations.

        :param X: features, *X* is converted into an array if *X* is a dataframe
        :return: predictions sorted for each observation
        """
        preds = self.predict_all(X)
        for i in range(preds.shape[0]):
            preds[i, :] = numpy.sort(preds[i, :])
        return preds
