"""
@file
@brief Metrics to compare machine learning.
"""
import numpy
from sklearn.metrics import r2_score

_known_functions = {
    'exp': numpy.exp,
    'log': numpy.log
}


def comparable_metric(metric_function, y_true, y_pred,
                      tr="log", inv_tr='exp', **kwargs):
    """
    Applies function on either the true target or/and the predictions
    before computing r2 score.

    :param metric_function: metric to compute
    :param y_true: expected targets
    :param y_pred: predictions
    :param sample_weight: weights
    :param multioutput: see :epkg:`sklearn:metrics:r2_score`
    :param tr: transformation applied on the target
    :param inv_tr: transformation applied on the predictions
    :return: results
    """
    tr = _known_functions.get(tr, tr)
    inv_tr = _known_functions.get(inv_tr, inv_tr)
    if tr is not None and not callable(tr):
        raise TypeError("Argument tr must be callable.")
    if inv_tr is not None and not callable(inv_tr):
        raise TypeError("Argument inv_tr must be callable.")
    if tr is None and inv_tr is None:
        raise ValueError(
            "tr and inv_tr cannot be both None at the same time.")
    if tr is None:
        return metric_function(y_true, inv_tr(y_pred), **kwargs)
    if inv_tr is None:
        return metric_function(tr(y_true), y_pred, **kwargs)
    return metric_function(tr(y_true), inv_tr(y_pred), **kwargs)


def r2_score_comparable(y_true, y_pred, *, sample_weight=None,
                        multioutput='uniform_average',
                        tr=None, inv_tr=None):
    """
    Applies function on either the true target or/and the predictions
    before computing r2 score.

    :param y_true: expected targets
    :param y_pred: predictions
    :param sample_weight: weights
    :param multioutput: see :epkg:`sklearn:metrics:r2_score`
    :param tr: transformation applied on the target
    :param inv_tr: transformation applied on the predictions
    :return: results

    Example:

    .. runpython::
        :showcode:

        import numpy
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from mlinsights.metrics import r2_score_comparable

        iris = datasets.load_iris()
        X = iris.data[:, :4]
        y = iris.target + 1

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model1 = LinearRegression().fit(X_train, y_train)
        print('r2', r2_score(y_test, model1.predict(X_test)))
        print('r2 log', r2_score(numpy.log(y_test), numpy.log(model1.predict(X_test))))
        print('r2 log comparable', r2_score_comparable(
            y_test, model1.predict(X_test), tr="log", inv_tr="log"))

        model2 = LinearRegression().fit(X_train, numpy.log(y_train))
        print('r2', r2_score(numpy.log(y_test), model2.predict(X_test)))
        print('r2 log', r2_score(y_test, numpy.exp(model2.predict(X_test))))
        print('r2 log comparable', r2_score_comparable(
            y_test, model2.predict(X_test), inv_tr="exp"))
    """
    return comparable_metric(r2_score, y_true, y_pred,
                             sample_weight=sample_weight,
                             multioutput=multioutput,
                             tr=tr, inv_tr=inv_tr)
