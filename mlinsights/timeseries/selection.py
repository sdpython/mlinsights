"""
@file
@brief Base class for timeseries.
"""
import numpy
from .base import BaseTimeSeries


def build_X_y(model, X, y, weights=None):
    """
    Builds standard *X, y* based in the given one.

    Parameters
    ----------

    model: a timeseries model (@see cl BaseTimeSeries)
    X: other times series, used as features, [n_obs, n_features],
        X may be empty (None)
    y: timeseries (one single vector),  [n_obs]
    weights: weights None or array [n_obs]

    Returns
    -------
    X: array of features [nrows, n_features + past]
        where `nrows = n_obs + model.delay2 - model.past + 2`
    y: array of targets [nrows]
    weights: None or array [nrows]

    A few examples.

    .. runpython::
        :showcode:

        import numpy
        from mlinsights.timeseries import build_X_y
        from mlinsights.timeseries.base import BaseTimeSeries

        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2)
        nx, ny, nw = build_X_y(bs, X, y, weights)
        print('X=', X)
        print('y=', y)
        print('nx=', nx)
        print('ny=', ny)

    With ``use_all_past=True``:

    .. runpython::
        :showcode:

        import numpy
        from mlinsights.timeseries.base import BaseTimeSeries

        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2, use_all_past=True)
        nx, ny, nw = build_X_y(bs, X, y, weights)
        print('X=', X)
        print('y=', y)
        print('nx=', nx)
        print('ny=', ny)
    """
    if not isinstance(model, BaseTimeSeries):
        raise TypeError(
            "model must be of type NaseTimeSeries not {}".format(type(model)))
    if model.use_all_past:
        ncol = X.shape[1] if X is not None else 0
        nrow = y.shape[0] - model.delay2 - model.past + 2
        new_X = numpy.empty(
            (nrow, ncol * model.past + model.past), dtype=y.dtype)
        if X is not None:
            for i in range(0, model.past):
                begin = i * ncol
                end = begin + ncol
                new_X[:, begin:end] = X[i: i + nrow]
        for i in range(0, model.past):
            end = y.shape[0] + i + model.delay1 - 1 - model.delay2
            new_X[:, i + ncol * model.past] = y[i: end]
        new_y = numpy.empty(
            (nrow, model.delay2 - model.delay1), dtype=y.dtype)
        for i in range(model.delay1, model.delay2):
            new_y[:, i - model.delay1] = y[i + 1:i + nrow + 1]
        new_weights = (None if weights is None
                       else weights[model.past - 1:model.past - 1 + nrow])
    else:
        ncol = X.shape[1] if X is not None else 0
        nrow = y.shape[0] - model.delay2 - model.past + 2
        new_X = numpy.empty((nrow, ncol + model.past), dtype=y.dtype)
        if X is not None:
            new_X[:, :X.shape[1]] = X[model.past -
                                      1: X.shape[0] - model.delay2 + 1]
        for i in range(model.past):
            end = y.shape[0] + i + model.delay1 - 1 - model.delay2
            new_X[:, i + ncol] = y[i: end]
        new_y = numpy.empty(
            (nrow, model.delay2 - model.delay1), dtype=y.dtype)
        for i in range(model.delay1, model.delay2):
            new_y[:, i - model.delay1] = y[i + 1:i + nrow + 1]
        new_weights = (None if weights is None
                       else weights[model.past - 1:model.past - 1 + nrow])
    return new_X, new_y, new_weights
