"""
@file
@brief Implements a predicatable *t-SNE*.
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
                 normalize=True, keep_tsne_outputs=False):
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
            raise AttributeError(  # pragma: no cover
                "normalizer {} does not have a 'transform' method.".format(
                    type(normalizer)))
        if not hasattr(transformer, "fit_transform"):
            raise AttributeError(  # pragma: no cover
                "transformer {} does not have a 'fit_transform' method.".format(
                    type(transformer)))
        if not hasattr(estimator, "predict"):
            raise AttributeError(  # pragma: no cover
                "estimator {} does not have a 'predict' method.".format(
                    type(estimator)))
        self.normalize = normalize

    def fit(self, X, y, sample_weight=None):
        """
        Trains a :epkg:`TSNE` then trains an estimator
        to approximate its outputs.

        :param X: numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        :param y: numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary
        :param sample_weight: numpy array of shape [n_samples]
            Individual weights for each sample
        :return: self, returns an instance of self.

        Fitted attributes:

        * `normalizer_`: trained normalier
        * `transformer_`: trained transformeer
        * `estimator_`: trained regressor
        * `tsne_outputs_`: t-SNE outputs if *keep_tsne_outputs* is True
        * `mean_`: average of the *t-SNE* output on each dimension
        * `inv_std_`: inverse of the standard deviation of the *t-SNE*
            output on each dimension
        * `loss_`: loss (:epkg:`sklearn:metrics:mean_squared_error`) between the predictions
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

        :param X: numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        :return: tranformed *X*
        """
        if self.normalizer_ is not None:
            X = self.normalizer_.transform(X)
        pred = self.estimator_.predict(X)
        if self.normalize:
            pred -= self.mean_
            pred *= self.inv_std_
        return pred
