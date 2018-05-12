"""
@file
@brief Featurizers for machine learned models.
"""
import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class FeaturizerTypeError(TypeError):
    """
    Unable to process a type.
    """
    pass


def model_featurizer(model, **params):
    """
    Converts a model into a function which converts
    a vector into features produced by the model.
    It can be the output itself or intermediate results.

    @param      model       model
    @param      params      additional parameters
    @return                 function
    """
    tried = []
    if isinstance(model, LogisticRegression):
        return model_featurizer_lr(model, **params)
    else:
        tried.append(LogisticRegression)
    if isinstance(model, RandomForestClassifier):
        return model_featurizer_rfc(model, **params)
    else:
        tried.append(RandomForestClassifier)
    if hasattr(model, "layers"):
        # It should be a keras model.
        return model_featurizer_keras(model, **params)
    else:
        tried.append("Keras")
    raise FeaturizerTypeError("Unable to process type '{0}', allowed:\n{1}".format(
        type(model), "\n".join(sorted(str(_) for _ in tried))))


def is_vector(X):
    """
    Tells if *X* is a vector.

    @param      X       vector
    @return             boolean
    """
    if isinstance(X, list):
        if len(X) == 0 or isinstance(X[0], (list, tuple)):
            return False
        else:
            return True
    if isinstance(X, numpy.ndarray):
        if len(X.shape) > 1 and X.shape[0] != 1:
            return False
        else:
            return True
    if isinstance(X, pandas.DataFrame):
        if len(X.shape) > 1 and X.shape[0] != 1:
            return False
        else:
            return True
    raise TypeError(
        "Unable to guess if X is a vector, type(X)={0}".format(type(X)))


def wrap_predict_sklearn(X, fct, many):
    """
    Checks types and dimension.
    Calls *fct* and returns the approriate type.
    A vector if *X* is a vector, the raw output
    otherwise.

    @param      X       vector or list
    @param      fct     function
    @param      many    many observations or just one
    """
    isv = is_vector(X)
    if many == isv:
        raise ValueError("Inconsistency X is a single vector, many is True")
    if isv:
        X = [X]
    y = fct(X)
    if isv:
        y = y.ravel()
    return y


def model_featurizer_lr(model):
    """
    Builds a featurizer from a :epkg:`scikit-learn:linear_model:LogisticRegression`.
    It returns a function which returns ``model.decision_function(X)``.

    @param      model       model to use to featurize a vector
    @return                 function
    """

    def feat(X, model, many):
        "wraps sklearn"
        return wrap_predict_sklearn(X, model.decision_function, many)

    return lambda X, many, model=model: feat(X, model, many)


def model_featurizer_rfc(model, output=True):
    """
    Builds a featurizer from a :epkg:`scikit-learn:ensemble:RandomForestClassifier`.
    It returns a function which returns the output of every tree
    (method *apply*).

    @param      model       model to use to featurize a vector
    @param      output      use output (``model.predict_proba(X)``)
                            or trees output (``model.apply(X)``)
    @return                 function
    """
    if output:
        def feat(X, model, many):
            "wraps sklearn"
            return wrap_predict_sklearn(X, model.predict_proba, many)

        return lambda X, many, model=model: feat(X, model, many)
    else:
        def feat(X, model, many):
            "wraps sklearn"
            return wrap_predict_sklearn(X, model.apply, many)

        return lambda X, many, model=model: feat(X, model, many)


def wrap_predict_keras(X, fct, many, shapes):
    """
    Checks types and dimension.
    Calls *fct* and returns the approriate type.
    A vector if *X* is a vector, the raw output
    otherwise.

    @param      X       vector or list
    @param      fct     function
    @param      many    many observations or just one
    @param      shapes  expected input shapes for the neural network
    """
    if many:
        y = [fct(X[i]).ravel() for i in range(X.shape[0])]
        return numpy.stack(y)
    else:
        if len(X.shape) == len(shapes):
            return fct(X).ravel()
        else:
            x = X[numpy.newaxis, :, :, :]
            return fct(x).ravel()


def model_featurizer_keras(model, layer=None):
    """
    Builds a featurizer from a :epkg:`keras` model
    It returns a function which returns the output of one
    particular layer.

    @param      model       model to use to featurize a vector
    @param      layer       number of layers to keep
    @return                 function

    See `About Keras models <https://keras.io/models/about-keras-models/>`_.
    """
    if layer is not None:
        output = model.layers[layer].output
        model = model.__class__(model.input, output)

    def feat(X, model, many, shapes):
        "wraps keras"
        return wrap_predict_keras(X, model.predict, many, shapes)

    return lambda X, many, model=model, shapes=model._feed_input_shapes[0]: feat(X, model, many, shapes)
