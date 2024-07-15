from sklearn.base import clone
from sklearn.pipeline import Pipeline, _fit_transform_one

try:
    from sklearn.utils._user_interface import _print_elapsed_time
except ImportError:
    from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from .cache_model import MLCache


class PipelineCache(Pipeline):
    """
    Same as :epkg:`sklearn:pipeline:Pipeline` but it can
    skip training if it detects a step was already trained
    the model was already trained accross
    even in a different pipeline.

    :param steps: list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    :param cache_name: name of the cache, if None, a new name is created
    :param verbose: boolean, optional
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    The attribute *named_steps* is a bunch object, a dictionary
    with attribute access Read-only attribute to access any step
    parameter by user given name. Keys are step names and values
    are steps parameters.
    """

    def __init__(self, steps, cache_name=None, verbose=False):
        self.cache_name = cache_name
        Pipeline.__init__(self, steps, memory=None, verbose=verbose)
        if cache_name is None:
            cache_name = "Pipeline%d" % id(self)
        self.cache_name = cache_name

    def _get_fit_params_steps(self, fit_params):
        fit_params_steps = {name: {} for name, step in self.steps if step is not None}

        for pname, pval in fit_params.items():
            if "__" not in pname:
                if not isinstance(pval, dict):
                    raise ValueError(
                        f"For scikit-learn < 0.23, "
                        f"Pipeline.fit does not accept the {pname} parameter. "
                        f"You can pass parameters to specific steps of your "
                        f"pipeline using the stepname__parameter format, e.g. "
                        f"`Pipeline.fit(X, y, logisticregression__sample_weight"
                        f"=sample_weight)`."
                    )
                else:
                    fit_params_steps[pname].update(pval)
            else:
                step, param = pname.split("__", 1)
                fit_params_steps[step][param] = pval
        return fit_params_steps

    def _fit(self, X, y=None, *args, **fit_params_steps):
        if "routed_params" in fit_params_steps:
            # scikit-learn>=1.4
            routed_params = fit_params_steps["routed_params"]
        elif len(args) == 1:
            # scikit-learn>=1.4
            routed_params = args[0]
        else:
            # scikit-learn<1.4
            routed_params = None
        self.steps = list(self.steps)
        self._validate_steps()
        memory = check_memory(self.memory)
        fit_transform_one_cached = memory.cache(_fit_transform_one)
        if not MLCache.has_cache(self.cache_name):
            self.cache_ = MLCache.create_cache(self.cache_name)
        else:
            self.cache_ = MLCache.get_cache(self.cache_name)
        Xt = X
        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            params = transformer.get_params()
            params["__class__"] = transformer.__class__.__name__
            params["X"] = Xt
            if (
                hasattr(transformer, "is_classifier") and transformer.is_classifier()
            ) or (hasattr(transformer, "is_regressor") and transformer.is_regressor()):
                params["y"] = y
            cached = self.cache_.get(params)
            if cached is None:
                cloned_transformer = clone(transformer)
                if routed_params is None:
                    Xt, fitted_transformer = fit_transform_one_cached(
                        cloned_transformer,
                        Xt,
                        y,
                        None,
                        message_clsname="PipelineCache",
                        message=self._log_message(step_idx),
                        **fit_params_steps[name],
                    )
                else:
                    Xt, fitted_transformer = fit_transform_one_cached(
                        cloned_transformer,
                        Xt,
                        y,
                        None,
                        message_clsname="PipelineCache",
                        message=self._log_message(step_idx),
                        params=routed_params[name],
                    )
                self.cache_.cache(params, fitted_transformer)
            else:
                fitted_transformer = cached
                Xt = fitted_transformer.transform(Xt)

            self.steps[step_idx] = (name, fitted_transformer)
        return Xt
