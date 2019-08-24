"""
@file
@brief Implements a slightly different
version of the :epkg:`sklearn:compose:TransformedTargetRegressor`.
"""
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from .sklearn_transform_inv import BaseReciprocalTransformer
from .sklearn_transform_inv_fct import FunctionReciprocalTransformer


class TransformedTargetRegressor2(BaseEstimator, RegressorMixin):
    """
    Meta-estimator to regress on a transformed target.
    Useful for applying a non-linear transformation in regression
    problems.

    Parameters
    ----------
    regressor : object, default=LinearRegression()
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
    transformer : str or object of type @see cl BaseReciprocalTransformer

    Attributes
    ----------
    regressor_ : object
        Fitted regressor.
    transformer_ : object
        Transformer used in ``fit`` and ``predict``.

    Examples
    --------

    .. runpython::
        :showcode:

        import numpy
        from sklearn.linear_model import LinearRegression
        from sklearn.compose import TransformedTargetRegressor

        tt = TransformedTargetRegressor2(regressor=LinearRegression(),
                                         transformer='log')
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        print(tt.fit(X, y))
        print(tt.score(X, y))
        print(tt.regressor_.coef_)
    """

    def __init__(self, regressor=None, transformer=None):
        self.regressor = regressor
        self.transformer = transformer

    def fit(self, X, y, sample_weight=None):
        """
        Fits the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
        """
        if isinstance(self.transformer, str):
            self.transformer_ = FunctionReciprocalTransformer(self.transformer)
        elif isinstance(self.transformer, BaseReciprocalTransformer):
            self.transformer_ = clone(self.transformer)
        else:
            raise TypeError("Transformer {} must be a string or on object of type "
                            "BaseReciprocalTransformer.".format(type(self.transformer)))

        self.transformer_.fit(X, y, sample_weight=sample_weight)
        X_trans, y_trans = self.transformer_.transform(X, y)

        if self.regressor is None:
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        if sample_weight is None:
            self.regressor_.fit(X_trans, y_trans)
        else:
            self.regressor_.fit(X_trans, y_trans, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """
        Predicts using the base regressor, applying inverse.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "regressor_")
        check_is_fitted(self, "transformer_")
        X_trans, _ = self.transformer_.transform(X, None)
        pred = self.regressor_.predict(X_trans)

        inv = self.transformer_.get_fct_inv()
        _, pred_inv = inv.transform(X_trans, pred)
        return pred_inv

    def score(self, X, y, sample_weight=None):
        """
        Scores the model with r2 metric
        :epkg:`sklearn:metrics:r2_score`.
        """
        yp = self.predict(X)
        return r2_score(y, yp, sample_weight=sample_weight)

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}
