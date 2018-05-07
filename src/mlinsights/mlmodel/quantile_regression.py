"""
@file
@brief Implements a quantile linear regression.
"""
import numpy
from sklearn.linear_model import LinearRegression


class QuantileLinearRegression(LinearRegression):
    """
    Quantile Linear Regression or linear regression
    trained with norm *L1*. This class inherits from
    :epkg:`sklearn:linear_models:LinearRegression`.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : array
        Independent term in the linear model.

    n_iter_: int
        Number of iterations at training time.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1, delta=0.0001, max_iter=10, verbose=False):
        """
        Parameters
        ----------
        fit_intercept: boolean, optional, default True
            whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations
            (e.g. data is expected to be already centered).

        normalize: boolean, optional, default False
            This parameter is ignored when ``fit_intercept`` is set to False.
            If True, the regressors X will be normalized before regression by
            subtracting the mean and dividing by the l2-norm.
            If you wish to standardize, please use
            :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
            an estimator with ``normalize=False``.

        copy_X: boolean, optional, default True
            If True, X will be copied; else, it may be overwritten.

        n_jobs: int, optional, default 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. This will only provide speedup for
            n_targets > 1 and sufficient large problems.

        max_iter: int, optional, default 1
            The number of iteration to do at training time.
            This parameter is specific to the quantile regression.

        delta: float, optional, default 0.0001
            Used to ensure matrices has an inverse
            (*M + delta*I*).

        verbose: bool, optional, default False
            Prints error at each iteration of the optimisation.
        """
        LinearRegression.__init__(self, fit_intercept=fit_intercept, normalize=normalize,
                                  copy_X=copy_X, n_jobs=n_jobs)
        self.max_iter = max_iter
        self.verbose = verbose
        self.delta = delta

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.
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

        Notes
        -----
        The implementation is not the most efficient
        as it calls multiple times method fit
        from :epkg:`sklearn:linear_models:LinearRegression`.
        Data gets checked and rescaled each time.
        The optimization follows the algorithm
        `Iteratively reweighted least squares <https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares>`_.
        """
        if len(y.shape) > 1 and y.shape[1] != 1:
            raise ValueError("QuantileLinearRegression only works for Y real")

        def compute_z(Xm, beta, Y, W, delta=0.0001):
            epsilon = numpy.abs(Y - Xm @ beta)
            r = numpy.reciprocal(numpy.maximum(
                epsilon, numpy.ones(epsilon.shape) * delta))
            r = r * r
            return r, epsilon

        if not isinstance(X, numpy.ndarray):
            if hasattr(X, 'as_matrix'):
                X = X.as_matrix()
            else:
                raise TypeError("X must be an array or a dataframe.")

        if self.fit_intercept:
            Xm = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        else:
            Xm = X

        clr = LinearRegression(fit_intercept=False, copy_X=self.copy_X,
                               n_jobs=self.n_jobs, normalize=self.normalize)

        W = numpy.ones(X.shape[0]) if sample_weight is None else sample_weight
        self.n_iter_ = 0
        lastE = None
        for i in range(0, self.max_iter):
            clr.fit(Xm, y, W)
            beta = clr.coef_
            W, epsilon = compute_z(Xm, beta, y, W, delta=self.delta)
            E = epsilon.sum()
            self.n_iter_ = i
            if self.verbose:
                print(
                    '[QuantileLinearRegression.fit] iter={0} error={1}'.format(i + 1, E))
            if lastE is not None and lastE == E:
                break
            lastE = E

        if self.fit_intercept:
            self.coef_ = beta[:-1]
            self.intercept_ = beta[-1]
        else:
            self.coef_ = beta
            self.intercept_ = 0

        return self
