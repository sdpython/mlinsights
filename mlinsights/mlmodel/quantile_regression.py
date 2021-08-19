# -*- coding: utf-8 -*-
"""
@file
@brief Implements a quantile linear regression.
"""
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


class QuantileLinearRegression(LinearRegression):
    """
    Quantile Linear Regression or linear regression
    trained with norm :epkg:`L1`. This class inherits from
    :epkg:`sklearn:linear_models:LinearRegression`.
    See notebook :ref:`quantileregressionrst`.

    Norm :epkg:`L1` is chosen if ``quantile=0.5``, otherwise,
    for *quantile=*:math:`\\rho`,
    the following error is optimized:

    .. math::

        \\sum_i \\rho |f(X_i) - Y_i|^- + (1-\\rho) |f(X_i) - Y_i|^+

    where :math:`|f(X_i) - Y_i|^-= \\max(Y_i - f(X_i), 0)` and
    :math:`|f(X_i) - Y_i|^+= \\max(f(X_i) - Y_i, 0)`.
    :math:`f(i)` is the prediction, :math:`Y_i` the expected
    value.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1, delta=0.0001, max_iter=10, quantile=0.5,
                 positive=False, verbose=False):
        """
        :param fit_intercept: boolean, optional, default True
            whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations
            (e.g. data is expected to be already centered).
        :param normalize: boolean, optional, default False
            This parameter is ignored when ``fit_intercept`` is set to False.
            If True, the regressors X will be normalized before regression by
            subtracting the mean and dividing by the l2-norm.
            If you wish to standardize, please use
            :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
            an estimator with ``normalize=False``.
        :param copy_X: boolean, optional, default True
            If True, X will be copied; else, it may be overwritten.
        :param n_jobs: int, optional, default 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. This will only provide speedup for
            n_targets > 1 and sufficient large problems.
        :param max_iter: int, optional, default 1
            The number of iteration to do at training time.
            This parameter is specific to the quantile regression.
        :param delta: float, optional, default 0.0001
            Used to ensure matrices has an inverse
            (*M + delta*I*).
        :param quantile: float, by default 0.5,
            determines which quantile to use
            to estimate the regression.
        :param positive: when set to True, forces the coefficients to be positive.
        :param verbose: bool, optional, default False
            Prints error at each iteration of the optimisation.
        """
        try:
            LinearRegression.__init__(
                self, fit_intercept=fit_intercept, normalize=normalize,
                copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        except TypeError:
            # scikit-learn<0.24
            LinearRegression.__init__(
                self, fit_intercept=fit_intercept, normalize=normalize,
                copy_X=copy_X, n_jobs=n_jobs)
        self.max_iter = max_iter
        self.verbose = verbose
        self.delta = delta
        self.quantile = quantile

    def fit(self, X, y, sample_weight=None):
        """
        Fits a linear model with :epkg:`L1` norm which
        is equivalent to a quantile regression.
        The implementation is not the most efficient
        as it calls multiple times method fit
        from :epkg:`sklearn:linear_models:LinearRegression`.
        Data gets checked and rescaled each time.
        The optimization follows the algorithm
        `Iteratively reweighted least squares
        <https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares>`_.
        It is described in French at
        `Régression quantile
        <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx/notebooks/td_note_2017_2.html>`_.

        :param X: numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        :param y: numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary
        :param sample_weight: numpy array of shape [n_samples]
            Individual weights for each sample
        :return: self, returns an instance of self.

        Fitted attributes:

        * `coef_`: array, shape (n_features, ) or (n_targets, n_features)
            Estimated coefficients for the linear regression problem.
            If multiple targets are passed during the fit (y 2D), this
            is a 2D array of shape (n_targets, n_features), while if only
            one target is passed, this is a 1D array of length n_features.
        *  `intercept_`: array
            Independent term in the linear model.
        * `n_iter_`: int
            Number of iterations at training time.
        """
        if len(y.shape) > 1 and y.shape[1] != 1:
            raise ValueError("QuantileLinearRegression only works for Y real")

        def compute_z(Xm, beta, Y, W, delta=0.0001):
            "compute z"
            deltas = numpy.ones(X.shape[0]) * delta
            epsilon, mult = QuantileLinearRegression._epsilon(
                Y, Xm @ beta, self.quantile)
            r = numpy.reciprocal(numpy.maximum(  # pylint: disable=E1111
                epsilon, deltas))  # pylint: disable=E1111
            if mult is not None:
                epsilon *= 1 - mult
                r *= 1 - mult
            return r, epsilon

        if not isinstance(X, numpy.ndarray):
            if hasattr(X, 'values'):
                X = X.values
            else:
                raise TypeError("X must be an array or a dataframe.")

        if self.fit_intercept:
            Xm = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        else:
            Xm = X

        try:
            clr = LinearRegression(fit_intercept=False, copy_X=self.copy_X,
                                   n_jobs=self.n_jobs, normalize=self.normalize,
                                   positive=self.positive)
        except AttributeError:
            # scikit-learn<0.24
            clr = LinearRegression(fit_intercept=False, copy_X=self.copy_X,
                                   n_jobs=self.n_jobs, normalize=self.normalize)

        W = numpy.ones(X.shape[0]) if sample_weight is None else sample_weight
        self.n_iter_ = 0
        lastE = None
        for i in range(0, self.max_iter):
            clr.fit(Xm, y, W)
            beta = clr.coef_
            W, epsilon = compute_z(Xm, beta, y, W, delta=self.delta)
            if sample_weight is not None:
                W *= sample_weight
                epsilon *= sample_weight
            E = epsilon.sum()
            self.n_iter_ = i
            if self.verbose:
                print(  # pragma: no cover
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

    @staticmethod
    def _epsilon(y_true, y_pred, quantile, sample_weight=None):
        diff = y_pred - y_true
        epsilon = numpy.abs(diff)
        if quantile != 0.5:
            sign = numpy.sign(diff)  # pylint: disable=E1111
            mult = numpy.ones(y_true.shape[0])
            mult[sign > 0] *= quantile  # pylint: disable=W0143
            mult[sign < 0] *= (1 - quantile)  # pylint: disable=W0143
        else:
            mult = None
        if sample_weight is not None:
            epsilon *= sample_weight
        return epsilon, mult

    def score(self, X, y, sample_weight=None):
        """
        Returns Mean absolute error regression loss.

        :param X: array-like, shape = (n_samples, n_features)
            Test samples.
        :param y: array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.
        :param sample_weight: array-like, shape = [n_samples], optional
            Sample weights.
        :return: score : float
            mean absolute error regression loss
        """
        pred = self.predict(X)

        if self.quantile != 0.5:
            epsilon, mult = QuantileLinearRegression._epsilon(
                y, pred, self.quantile, sample_weight)
            if mult is not None:
                epsilon *= mult * 2
            return epsilon.sum() / X.shape[0]
        return mean_absolute_error(y, pred, sample_weight=sample_weight)
