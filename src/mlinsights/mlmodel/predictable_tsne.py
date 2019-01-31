"""
@file
@brief Implements a quantile linear regression.
"""
import inspect
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor


class PredictableTSNE(BaseEstimator, TransformerMixin):
    """
    :epkg:`t-SNE` is an interesting
    transform which can only be used to study data as there is no
    way to reproduce the result once it was fitted. That's why
    the class :epkg:`TSNE` does not have any method *transform*, only
    `fit_transform <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE.fit_transform>`_.
    This example proposes a way to train a machine learned model
    which approximates the outputs of a :epkg:`TSNE` transformer.
    Notebooks :ref:`predictabletsnerst` gives an example on how to
    use this class.
    """

    def __init__(self, normalizer=None, transformer=None, estimator=None, **kwargs):
        """
        @param      normalizer      None by default
        @param      transformer     :epkg:`sklearn:manifold:TSNE`
                                    by default
        @param      estimator       :epkg:`sklearn:neural_network:MLPRegressor`
                                    by default
        @param      kwargs          sent to :meth:`set_params
                                    <mlinsights.mlmodel.tsne_transformer.PredictableTSNE.set_params>`,
                                    see its documentation to understand how to specify parameters
        """
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)
        if estimator is None:
            estimator = MLPRegressor()
        if transformer is None:
            transformer = TSNE()
        self.estimator = estimator
        self.transformer = transformer
        self.normalizer = normalizer
        if normalizer is not None and not hasattr(normalizer, "transform"):
            raise AttributeError(
                "normalizer {} does not have a 'transform' method.".format(type(normalizer)))
        if not hasattr(transformer, "fit_transform"):
            raise AttributeError(
                "transformer {} does not have a 'fit_transform' method.".format(type(transformer)))
        if not hasattr(estimator, "predict"):
            raise AttributeError(
                "estimator {} does not have a 'predict' method.".format(type(estimator)))
        if kwargs:
            self.set_params(**kwargs)

    def fit(self, X, y, sample_weight=None, memoize_targets=None):
        """
        Runs a *k-means* on each class
        then trains a classifier on the
        extended set of features.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample

        memoize_targets: if not None, raw outputs of
            :epkg:`TSNE` is added to this list

        Returns
        -------
        self : returns an instance of self.

        Attributes
        ----------

        normalizer_: trained normalier

        transformer_: trained transformeer

        estimator_: trained regressor
        """
        params = dict(y=y, sample_weight=sample_weight)

        if self.normalizer is not None:
            sig = inspect.signature(self.normalizer.transform)
            pars = {}
            for p in ['sample_weight', 'y']:
                if p in sig.parameters and p in params:
                    pars[p] = params[p]
            self.normalizer_ = clone(self.normalizer).fit(X, **pars)
            X = self.normalizer_.transform(X)
        else:
            self.normalizer_ = None

        self.transformer_ = clone(self.transformer)

        sig = inspect.signature(self.transformer.fit_transform)
        pars = {}
        for p in ['sample_weight', 'y']:
            if p in sig.parameters and p in params:
                pars[p] = params[p]
        target = self.transformer_.fit_transform(X, **pars)

        if memoize_targets is not None:
            if not isinstance(memoize_targets, list):
                raise TypeError("memoize_targets must be a list")
            memoize_targets.append(target)

        sig = inspect.signature(self.estimator.fit)
        if 'sample_weight' in sig.parameters:
            self.estimator_ = clone(self.estimator).fit(
                X, target, sample_weight=sample_weight)
        else:
            self.estimator_ = clone(self.estimator).fit(X, target)

        return self

    def transform(self, X):
        """
        Runs the predictions.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        Returns
        -------
        tranformed *X*
        """
        if self.normalizer_ is not None:
            X = self.normalizer_.transform(X)
        return self.estimator_.predict(X)

    def get_params(self, deep=True):
        """
        Returns the parameters for all the embedded objects.

        @param      deep        unused here
        @return                 dict

        :meth:`set_params <mlinsights.mlmodel.tsne_transformer.PredictableTSNE.set_params>`
        describes the pattern parameters names follow.
        """
        res = {}
        if self.normalizer is not None:
            for k, v in self.normalizer.get_params().items():
                res["n_" + k] = v
        for k, v in self.transformer.get_params().items():
            res["t_" + k] = v
        for k, v in self.estimator.get_params().items():
            res["e_" + k] = v
        return res

    def set_params(self, **values):
        """
        Sets the parameters before training.
        Every parameter prefixed by ``'e_'`` is an estimator
        parameter, every parameter prefixed by ``'n_'`` is for
        a normalizer parameter, every parameter prefixed by
        ``t_`` is for a transformer parameter.

        @param      values      valeurs
        @return                 dict
        """
        pt, pe, pn = {}, {}, {}
        for k, v in values.items():
            if k.startswith('e_'):
                pe[k[2:]] = v
            elif k.startswith('t_'):
                pt[k[2:]] = v
            elif k.startswith('n_'):
                pn[k[2:]] = v
            else:
                raise ValueError("Unexpected parameter name '{0}'".format(k))
        self.transformer.set_params(**pt)
        self.estimator.set_params(**pe)
        if self.normalizer is not None:
            self.transformer.set_params(**pn)
        elif pn and self.normalizer is None:
            raise ValueError(
                "There is no normalizer, cannot change parameter {}.".format(pn))
