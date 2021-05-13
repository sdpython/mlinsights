"""
@file
@brief Implements new features such as polynomial features.
"""
import numpy
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from ._extended_features_polynomial import _transform_iall, _transform_ionly, _combinations_poly


class ExtendedFeatures(BaseEstimator, TransformerMixin):
    """
    Generates extended features such as polynomial features.

    :param kind: string
        ``'poly'`` for polynomial features,
        ``'poly-slow'`` for polynomial features in *scikit-learn 0.20.2*
    :param poly_degree: integer
        The degree of the polynomial features. Default = 2.
    :param poly_interaction_only: boolean
        If true, only interaction features are produced: features that
        are products of at most degree distinct input features
        (so not ``x[1] ** 2, x[0] * x[2] ** 3``, etc.).
    :param poly_include_bias: boolean
        If True (default), then include a bias column, the feature in
        which all polynomial powers are zero (i.e. a column of ones -
        acts as an intercept term in a linear model).

    Fitted attributes:

    * `n_input_features_`: int
        The total number of input features.
    * `n_output_features_`: int
        The total number of polynomial output features. The number of output
        features is computed by iterating over all suitably sized combinations
        of input features.
    """

    def __init__(self, kind='poly', poly_degree=2, poly_interaction_only=False,
                 poly_include_bias=True):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.kind = kind
        self.poly_degree = poly_degree
        self.poly_include_bias = poly_include_bias
        self.poly_interaction_only = poly_interaction_only

    def get_feature_names(self, input_features=None):
        """
        Returns feature names for output features.

        :param input_features: list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.
        :return: output_feature_names : list of string, length n_output_features
        """
        if self.kind == 'poly':
            return self._get_feature_names_poly(input_features)
        if self.kind == 'poly-slow':
            return self._get_feature_names_poly(input_features)
        raise ValueError(  # pragma: no cover
            "Unknown extended features '{}'.".format(self.kind))

    def _get_feature_names_poly(self, input_features=None):
        """
        Returns feature names for output features for
        the polynomial features.
        """
        if input_features is None:
            input_features = ["x%d" %
                              i for i in range(0, self.n_input_features_)]
        elif len(input_features) != self.n_input_features_:
            raise ValueError(  # pragma: no cover
                "input_features should contain {} strings.".format(
                    self.n_input_features_))

        names = ["1"] if self.poly_include_bias else []
        n = self.n_input_features_
        interaction_only = self.poly_interaction_only
        for d in range(0, self.poly_degree):
            if d == 0:
                pos = len(names)
                names.extend(input_features)
                index = list(range(pos, len(names)))
                index.append(len(names))
            else:
                new_index = []
                end = index[-1]
                for i in range(0, n):
                    a = index[i]
                    new_index.append(len(names))
                    start = a + (index[i + 1] - index[i]
                                 if interaction_only else 0)
                    names.extend([a + " " + input_features[i]
                                  for a in names[start:end]])
                new_index.append(len(names))
                index = new_index

        def process_name(col):
            scol = col.split()
            res = []
            for c in sorted(scol):
                if len(res) == 0 or res[-1][0] != c:
                    res.append((c, 1))
                else:
                    res[-1] = (c, res[-1][1] + 1)
            return " ".join(["%s^%d" % r if r[1] > 1 else r[0] for r in res])

        names = [process_name(s) for s in names]
        return names

    def fit(self, X, y=None):
        """
        Compute number of output features.

        :param X: array-like, shape (n_samples, n_features)
            The data.
        :return: self : instance
        """
        self.n_input_features_ = X.shape[1]
        self.n_output_features_ = len(self.get_feature_names())

        if self.kind == 'poly':
            return self._fit_poly(X, y)
        elif self.kind == 'poly-slow':
            return self._fit_poly(X, y)
        raise ValueError(  # pragma: no cover
            "Unknown extended features '{}'.".format(self.kind))

    def _fit_poly(self, X, y=None):
        """
        Fitting method for the polynomial features.
        """
        check_array(X, accept_sparse=False)
        return self

    def transform(self, X):
        """
        Transforms data to extended features.

        :param X: array-like, shape [n_samples, n_features]
            The data to transform, row by row.
            rns
        :param XP: numpy.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of polynomial
            features generated from the combination of inputs.
        """
        n_features = X.shape[1]
        if n_features != self.n_input_features_:
            raise ValueError(  # pragma: no cover
                "X shape does not match training shape")
        if self.kind == 'poly':
            return self._transform_poly(X)
        if self.kind == 'poly-slow':
            return self._transform_poly_slow(X)
        raise ValueError(  # pragma: no cover
            "Unknown extended features '{}'.".format(self.kind))

    def _transform_poly(self, X):
        """
        Transforms data to polynomial features.
        """
        if sparse.isspmatrix(X):
            raise NotImplementedError(  # pragma: no cover
                "Not implemented for sparse matrices.")

        XP = numpy.empty(
            (X.shape[0], self.n_output_features_), dtype=X.dtype)

        def multiply(A, B, C):
            return numpy.multiply(A, B, out=C)

        def final(X):
            return X

        if self.poly_interaction_only:
            return _transform_ionly(self.poly_degree, self.poly_include_bias,
                                    XP, X, multiply, final)
        return _transform_iall(self.poly_degree, self.poly_include_bias,
                               XP, X, multiply, final)

    def _transform_poly_slow(self, X):
        """
        Transforms data to polynomial features.
        """
        if sparse.isspmatrix(X):
            raise NotImplementedError(  # pragma: no cover
                "Not implemented for sparse matrices.")

        comb = _combinations_poly(X.shape[1], self.poly_degree, self.poly_interaction_only,
                                  include_bias=self.poly_include_bias)
        order = 'C'  # how to get order from X.
        XP = numpy.empty((X.shape[0], self.n_output_features_),
                         dtype=X.dtype, order=order)
        for i, comb in enumerate(comb):
            XP[:, i] = X[:, comb].prod(1)
        return XP
