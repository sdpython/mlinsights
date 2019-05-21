"""
@file
@brief Caches training.
"""
from sklearn.base import clone
from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.utils import _print_elapsed_time
from .cache_model import MLCache


class PipelineCache(Pipeline):
    """
    Same as :epkg:`sklearn:pipeline:Pipeline` but it can
    skip training if it detects a step was already trained
    the model was already trained accross
    even in a different pipeline.

    Parameters
    ----------

    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    cache_name : name of the cache, if None, a new name is created
    verbose : boolean, optional
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    def __init__(self, steps, cache_name=None, verbose=False):
        Pipeline.__init__(self, steps, memory=None, verbose=verbose)
        if cache_name is None:
            cache_name = "Pipeline%d" % id(self)
        self.cache_name = cache_name

    def _get_fit_params_steps(self, fit_params):
        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}

        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname))
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps

    def _fit(self, X, y=None, **fit_params):

        self.steps = list(self.steps)
        self._validate_steps()
        fit_params_steps = self._get_fit_params_steps(fit_params)
        if not MLCache.has_cache(self.cache_name):
            self.cache_ = MLCache.create_cache(self.cache_name)
        else:
            self.cache_ = MLCache.get_cache(self.cache_name)

        Xt = X
        for (step_idx, name, transformer) in self._iter(with_final=False, filter_passthrough=False):
            if (transformer is None or transformer == 'passthrough'):
                with _print_elapsed_time('Pipeline', self._log_message(step_idx)):
                    continue

            params = transformer.get_params()
            params['X'] = X
            params['y'] = y
            cached = self.cache_.get(params)
            if cached is None:
                cloned_transformer = clone(transformer)
                Xt, fitted_transformer = _fit_transform_one(
                    cloned_transformer, Xt, y, None,
                    message_clsname='PipelineCache',
                    message=self._log_message(step_idx),
                    **fit_params_steps[name])
                self.cache_.cache(params, fitted_transformer)
            else:
                fitted_transformer = cached

            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return Xt, {}
        return Xt, fit_params_steps[self.steps[-1][0]]
