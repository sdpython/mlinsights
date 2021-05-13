"""
@file
@brief Implements a slightly different
version of the :epkg:`sklearn:compose:TransformedTargetRegressor`.
"""
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from .sklearn_transform_inv import BaseReciprocalTransformer
from .sklearn_transform_inv_fct import FunctionReciprocalTransformer, PermutationReciprocalTransformer


def _common_get_transform(transformer, is_regression):
    if isinstance(transformer, str):
        closest = is_regression
        if transformer == 'permute':
            return PermutationReciprocalTransformer(closest=closest)
        else:
            return FunctionReciprocalTransformer(transformer)
    elif isinstance(transformer, BaseReciprocalTransformer):
        return clone(transformer)
    raise TypeError("Transformer {} must be a string or on object of type "
                    "BaseReciprocalTransformer.".format(type(transformer)))


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
        from mlinsights.mlmodel import TransformedTargetRegressor2

        tt = TransformedTargetRegressor2(regressor=LinearRegression(),
                                         transformer='log')
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.exp(2 * X).ravel()
        print(tt.fit(X, y))
        print(tt.score(X, y))
        print(tt.regressor_.coef_)

    See notebook :ref:`sklearntransformedtargetrst` for a more complete example.
    """

    def __init__(self, regressor=None, transformer=None):
        self.regressor = regressor
        self.transformer = transformer

    def fit(self, X, y, sample_weight=None):
        """
        Fits the model according to the given training data.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        :param y: array-like, shape (n_samples,)
            Target values.
        :param sample_weight: array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        :return: self, object
        """
        self.transformer_ = _common_get_transform(self.transformer, True)
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

        :param X: {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        :return: y_hat : array, shape = (n_samples,)
            Predicted values.
        """
        if not hasattr(self, 'regressor_'):
            raise NotFittedError(  # pragma: no cover
                "This instance {} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method.".format(
                    type(self)))
        X_trans, _ = self.transformer_.transform(X, None)
        pred = self.regressor_.predict(X_trans)

        inv = self.transformer_.get_fct_inv()
        _, pred_inv = inv.transform(X_trans, pred)
        return pred_inv

    def score(self, X, y, sample_weight=None):
        """
        Scores the model with
        :epkg:`sklearn:metrics:r2_score`.
        """
        yp = self.predict(X)
        return r2_score(y, yp, sample_weight=sample_weight)

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}


class TransformedTargetClassifier2(BaseEstimator, ClassifierMixin):
    """
    Meta-estimator to classify on a transformed target.
    Useful for applying permutation transformation in classification
    problems.

    Parameters
    ----------
    classifier : object, default=LogisticRegression()
        Classifier object such as derived from ``ClassifierMixin``. This
        classifier will automatically be cloned each time prior to fitting.
    transformer : str or object of type @see cl BaseReciprocalTransformer

    Attributes
    ----------
    classifier_ : object
        Fitted classifier.
    transformer_ : object
        Transformer used in ``fit``, ``predict``, ``decision_function``,
        ``predict_proba``.

    Examples
    --------

    .. runpython::
        :showcode:

        import numpy
        from sklearn.linear_model import LogisticRegression
        from mlinsights.mlmodel import TransformedTargetClassifier2

        tt = TransformedTargetClassifier2(classifier=LogisticRegression(),
                                         transformer='permute')
        X = numpy.arange(4).reshape(-1, 1)
        y = numpy.array([0, 1, 0, 1])
        print(tt.fit(X, y))
        print(tt.score(X, y))
        print(tt.classifier_.coef_)

    See notebook :ref:`sklearntransformedtargetrst` for a more complete example.
    """

    def __init__(self, classifier=None, transformer=None):
        self.classifier = classifier
        self.transformer = transformer

    def fit(self, X, y, sample_weight=None):
        """
        Fits the model according to the given training data.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        :param y: array-like, shape (n_samples,)
            Target values.
        :param sample_weight: array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        :return: self, object
        """
        self.transformer_ = _common_get_transform(self.transformer, False)
        self.transformer_.fit(X, y, sample_weight=sample_weight)
        X_trans, y_trans = self.transformer_.transform(X, y)

        if self.classifier is None:
            self.classifier_ = LogisticRegression()
        else:
            self.classifier_ = clone(self.classifier)

        if sample_weight is None:
            self.classifier_.fit(X_trans, y_trans)
        else:
            self.classifier_.fit(X_trans, y_trans, sample_weight=sample_weight)

        return self

    def _check_is_fitted(self):
        if not hasattr(self, 'classifier_'):
            raise NotFittedError(
                "This instance {} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method.".format(
                    type(self)))

    @property
    def classes_(self):
        """
        Returns the classes.
        """
        self._check_is_fitted()
        inv = self.transformer_.get_fct_inv()
        _, pred_inv = inv.transform(None, self.classifier_.classes_)
        return pred_inv

    def _apply(self, X, method):
        """
        Calls *predict*, *predict_proba* or *decision_function*
        using the base classifier, applying inverse.

        :param X: {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        :return: y_hat, array, shape = (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        if not hasattr(self.classifier_, method):
            raise RuntimeError("Unable to find method '{}' in model {}.".format(
                method, type(self.classifier_)))
        meth = getattr(self.classifier_, method)
        X_trans, _ = self.transformer_.transform(X, None)
        pred = meth(X_trans)
        inv = self.transformer_.get_fct_inv()
        _, pred_inv = inv.transform(X_trans, pred)
        return pred_inv

    def predict(self, X):
        """
        Predicts using the base classifier, applying inverse.

        :param X: {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        :return: y_hat, array, shape = (n_samples,)
            Predicted values.
        """
        return self._apply(X, 'predict')

    def predict_proba(self, X):
        """
        Predicts using the base classifier, applying inverse.

        :param X: {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        :return: predict probabilities, array, shape = (n_samples, n_classes)
            Predicted values.
        """
        return self._apply(X, 'predict_proba')

    def decision_function(self, X):
        """
        Predicts using the base classifier, applying inverse.

        :param X: {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        :return: raw score : array, shape = (n_samples, ?)
        """
        return self._apply(X, 'decision_function')

    def score(self, X, y, sample_weight=None):
        """
        Scores the model with
        :epkg:`sklearn:metrics:accuracy_score`.
        """
        yp = self.predict(X)
        return accuracy_score(y, yp, sample_weight=sample_weight)

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}
