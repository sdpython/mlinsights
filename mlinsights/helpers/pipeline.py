import textwrap
import warnings
from types import MethodType
from sklearn.base import (
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
    BaseEstimator,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor


def enumerate_pipeline_models(pipe, coor=None, vs=None):
    """
    Enumerates all the models within a pipeline.

    :param pipe: *scikit-learn* pipeline
    :param coor: current coordinate
    :param vs: subset of variables for the model, None for all
    :return: iterator on models ``tuple(coordinate, model)``

    See example :ref:`l-visualize-pipeline-example`.
    """
    if coor is None:
        coor = (0,)
    if pipe == "passthrough":

        class PassThrough:
            "dummy class to help display"
            pass

        yield coor, PassThrough(), vs
    else:
        yield coor, pipe, vs
        if hasattr(pipe, "transformer_and_mapper_list") and len(
            pipe.transformer_and_mapper_list
        ):
            # azureml DataTransformer
            raise NotImplementedError("Unable to handle this specific case.")
        elif hasattr(pipe, "mapper") and pipe.mapper:
            # azureml DataTransformer
            for couple in enumerate_pipeline_models(pipe.mapper, coor + (0,)):
                yield couple
        elif hasattr(pipe, "built_features"):
            # sklearn_pandas.dataframe_mapper.DataFrameMapper
            for i, (columns, transformers, _) in enumerate(pipe.built_features):
                if isinstance(columns, str):
                    columns = (columns,)
                if transformers is None:
                    yield (coor + (i,)), None, columns
                else:
                    for couple in enumerate_pipeline_models(
                        transformers, coor + (i,), columns
                    ):
                        yield couple
        elif isinstance(pipe, Pipeline):
            for i, (_, model) in enumerate(pipe.steps):
                for couple in enumerate_pipeline_models(model, coor + (i,)):
                    yield couple
        elif isinstance(pipe, ColumnTransformer):
            for i, (_, fitted_transformer, column) in enumerate(pipe.transformers):
                for couple in enumerate_pipeline_models(
                    fitted_transformer, coor + (i,), column
                ):
                    yield couple
        elif isinstance(pipe, FeatureUnion):
            for i, (_, model) in enumerate(pipe.transformer_list):
                for couple in enumerate_pipeline_models(model, coor + (i,)):
                    yield couple
        elif isinstance(pipe, TransformedTargetRegressor):
            raise NotImplementedError(
                "Not yet implemented for TransformedTargetRegressor."
            )
        elif isinstance(pipe, (TransformerMixin, ClassifierMixin, RegressorMixin)):
            pass
        elif isinstance(pipe, BaseEstimator):
            pass
        else:
            raise TypeError(f"pipe is not a scikit-learn object: {type(pipe)}\n{pipe}")


class BaseEstimatorDebugInformation:
    """
    Stores information when the outputs of a pipeline
    is computed. It as added by function
    :func:`alter_pipeline_for_debugging`.
    """

    def __init__(self, model):
        self.model = model
        self.inputs = {}
        self.outputs = {}
        self.methods = {}
        if hasattr(model, "transform") and callable(model.transform):
            model._debug_transform = model.transform
            self.methods["transform"] = lambda model, X: model._debug_transform(X)
        if hasattr(model, "predict") and callable(model.predict):
            model._debug_predict = model.predict
            self.methods["predict"] = lambda model, X: model._debug_predict(X)
        if hasattr(model, "predict_proba") and callable(model.predict_proba):
            model._debug_predict_proba = model.predict_proba
            self.methods["predict_proba"] = lambda model, X: model._debug_predict_proba(
                X
            )
        if hasattr(model, "decision_function") and callable(model.decision_function):
            model._debug_decision_function = model.decision_function
            self.methods[
                "decision_function"
            ] = lambda model, X: model._debug_decision_function(X)

    def __repr__(self):
        """
        usual
        """
        return self.to_str()

    def to_str(self, nrows=5):
        """
        Tries to produce a readable message.
        """
        rows = [f"BaseEstimatorDebugInformation({self.model.__class__.__name__})"]
        for k in sorted(self.inputs):
            if k in self.outputs:
                rows.append("  " + k + "(")
                self.display(self.inputs[k], nrows)
                rows.append(textwrap.indent(self.display(self.inputs[k], nrows), "   "))
                rows.append("  ) -> (")
                rows.append(
                    textwrap.indent(self.display(self.outputs[k], nrows), "   ")
                )
                rows.append("  )")
            else:
                raise KeyError(f"Unable to find output for method '{k}'.")
        return "\n".join(rows)

    def display(self, data, nrows):
        """
        Displays the first
        """
        text = str(data)
        rows = text.split("\n")
        if len(rows) > nrows:
            rows = rows[:nrows]
            rows.append("...")
        if hasattr(data, "shape"):
            rows.insert(0, f"shape={data.shape!r} type={type(data)!r}")
        else:
            rows.insert(0, f"type={type(data)!r}")
        return "\n".join(rows)


def alter_pipeline_for_debugging(pipe):
    """
    Overwrite methods *transform*, *predict*, *predict_proba*
    or *decision_function* to collect the last inputs and outputs
    seen in these methods.

    :param pipe: *scikit-learn* pipeline

    The object *pipe* is modified, it should be copied
    before calling this function if you need the object
    untouched after that. The prediction is slower.
    See notebook :ref:`l-visualize-pipeline-example`.
    """

    def transform(self, X, *args, **kwargs):
        self._debug.inputs["transform"] = X
        y = self._debug.methods["transform"](self, X, *args, **kwargs)
        self._debug.outputs["transform"] = y
        return y

    def predict(self, X, *args, **kwargs):
        self._debug.inputs["predict"] = X
        y = self._debug.methods["predict"](self, X, *args, **kwargs)
        self._debug.outputs["predict"] = y
        return y

    def predict_proba(self, X, *args, **kwargs):
        self._debug.inputs["predict_proba"] = X
        y = self._debug.methods["predict_proba"](self, X, *args, **kwargs)
        self._debug.outputs["predict_proba"] = y
        return y

    def decision_function(self, X, *args, **kwargs):
        self._debug.inputs["decision_function"] = X
        y = self._debug.methods["decision_function"](self, X, *args, **kwargs)
        self._debug.outputs["decision_function"] = y
        return y

    new_methods = {
        "decision_function": decision_function,
        "transform": transform,
        "predict": predict,
        "predict_proba": predict_proba,
    }

    if hasattr(pipe, "_debug"):
        raise RuntimeError(
            "The same operator cannot be used twice in "
            "the same pipeline or this method was called "
            "a second time."
        )

    for model_ in enumerate_pipeline_models(pipe):
        model = model_[1]
        model._debug = BaseEstimatorDebugInformation(model)
        for k in model._debug.methods:
            try:
                setattr(model, k, MethodType(new_methods[k], model))
            except AttributeError:
                warnings.warn(
                    f"Unable to overwrite method {k!r} for class {type(model)!r}."
                )
