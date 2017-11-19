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


def model_featurizer(model):
    """
    Converts a model into a function which converts
    a vector into features produced by the model.
    It can be the output itself or intermediate results.

    @param      model       model
    @return                 function
    """
    tried = []
    if isinstance(model, LogisticRegression):
        return model_featurizer_lr(model)
    else:
        tried.append(LogisticRegression)
    if isinstance(model, RandomForestClassifier):
        return model_featurizer_rfc(model)
    else:
        tried.append(RandomForestClassifier)
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
        if (len(X.shape) > 1 and X.shape[0] != 1):
            return False
        else:
            return True
    if isinstance(X, pandas.DataFrame):
        if (len(X.shape) > 1 and X.shape[0] != 1):
            return False
        else:
            return True
    raise TypeError(
        "Unable to guess if X is a vector, type(X)={0}".format(type(X)))


def wrap_predict(X, fct):
    """
    Checks types and dimension.
    Calls *fct* and returns the approriate type.
    A vector if *X* is a vector, the raw output
    otherwise.

    @param      X       vector or list
    @param      fct     function
    """
    isv = is_vector(X)
    if isv:
        X = [X]
    y = fct(X)
    if isv:
        y = y.ravel()
    return y


def model_featurizer_lr(model):
    """
    Build a featurizer from a :epkg:`scikit-learn:linear_model:LogisticRegresion`.
    It returns a function which returns ``model.decision_function(X)``.
    """

    def feat(X, model):
        return wrap_predict(X, model.decision_function)

    return lambda X, model=model: feat(X, model)


def model_featurizer_rfc(model):
    """
    Build a featurizer from a :epkg:`scikit-learn:ensemble:RandomForestClassifier`.
    It returns a function which returns the output of every tree
    (method *apply*).
    """

    def feat(X, model):
        return wrap_predict(X, model.apply)

    return lambda X, model=model: feat(X, model)
