"""
@file
@brief Implements a quantile linear regression.
"""
import inspect
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


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

    def __init__(self, normalizer=None, transformer=None, estimator=None,
                 normalize=True, keep_tsne_outputs=False, **kwargs):
        """
        @param      normalizer          None by default
        @param      transformer         :epkg:`sklearn:manifold:TSNE`
                                        by default
        @param      estimator           :epkg:`sklearn:neural_network:MLPRegressor`
                                        by default
        @param      normalize           normalizes the outputs, centers and normalizes
                                        the output of the *t-SNE* and applies that same
                                        normalization to he prediction of the estimator
        @param      keep_tsne_output    if True, keep raw outputs of
                                        :epkg:`TSNE` is stored in member
                                        *tsne_outputs_*
        @param      kwargs              sent to :meth:`set_params
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
        self.keep_tsne_outputs = keep_tsne_outputs
        if normalizer is not None and not hasattr(normalizer, "transform"):
            raise AttributeError(
                "normalizer {} does not have a 'transform' method.".format(type(normalizer)))
        if not hasattr(transformer, "fit_transform"):
            raise AttributeError(
                "transformer {} does not have a 'fit_transform' method.".format(type(transformer)))
        if not hasattr(estimator, "predict"):
            raise AttributeError(
                "estimator {} does not have a 'predict' method.".format(type(estimator)))
        self.normalize = normalize
        if kwargs:
            self.set_params(**kwargs)

    def fit(self, X, y, sample_weight=None):
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

        Returns
        -------
        self : returns an instance of self.

        Attributes
        ----------

        normalizer_: trained normalier

        transformer_: trained transformeer

        estimator_: trained regressor

        tsne_outputs_: t-SNE outputs if *keep_tsne_outputs* is True

        mean_: average of the *t-SNE* output on each dimension

        inv_std_: inverse of the standard deviation of the *t-SNE*
            output on each dimension

        loss_: loss (:epkg:`sklearn:metrics:mean_squared_error`) between the predictions
            and the outputs of t-SNE
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

        sig = inspect.signature(self.estimator.fit)
        if 'sample_weight' in sig.parameters:
            self.estimator_ = clone(self.estimator).fit(
                X, target, sample_weight=sample_weight)
        else:
            self.estimator_ = clone(self.estimator).fit(X, target)
        mean = target.mean(axis=0)
        var = target.std(axis=0)
        self.mean_ = mean
        self.inv_std_ = 1. / var
        exp = (target - mean) * self.inv_std_
        got = (self.estimator_.predict(X) - mean) * self.inv_std_
        self.loss_ = mean_squared_error(exp, got)
        if self.keep_tsne_outputs:
            self.tsne_outputs_ = exp if self.normalize else target
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
        pred = self.estimator_.predict(X)
        if self.normalize:
            pred -= self.mean_
            pred *= self.inv_std_
        return pred

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
            self.normalizer.set_params(**pn)
        elif pn and self.normalizer is None:
            raise ValueError(
                "There is no normalizer, cannot change parameter {}.".format(pn))
