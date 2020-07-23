"""
@file
@brief Timeseries data manipulations.
"""
import numpy
from sklearn import get_config


def build_ts_X_y(model, X, y, weights=None, same_rows=False):
    """
    Builds standard *X, y* based in the given one.

    @param      model       a timeseries model (@see cl BaseTimeSeries)
    @param      X           times series, used as features, [n_obs, n_features],
                            X may be empty (None)
    @param      y           timeseries (one single vector),  [n_obs]
    @param      weights     weights None or array [n_obs]
    @param      same_rows   keep the same number of rows
                            as the original datasets, use nan when no value is
                            available
    @return                 *(X, y, weights)*:  X is array of features [nrows, n_features + past]
                            where `nrows = n_obs + model.delay2 - model.past + 2`,
                            y is an array of targets [nrows],
                            weights is None or array [nrows]

    .. runpython::
        :showcode:

        import numpy
        from mlinsights.timeseries import build_ts_X_y
        from mlinsights.timeseries.base import BaseTimeSeries

        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        print('X=', X)
        print('y=', y)
        print('nx=', nx)
        print('ny=', ny)

    With ``use_all_past=True``:

    .. runpython::
        :showcode:

        import numpy
        from mlinsights.timeseries.base import BaseTimeSeries
        from mlinsights.timeseries import build_ts_X_y

        X = numpy.arange(10).reshape(5, 2)
        y = numpy.arange(5) * 100
        weights = numpy.arange(5) * 1000
        bs = BaseTimeSeries(past=2, use_all_past=True)
        nx, ny, nw = build_ts_X_y(bs, X, y, weights)
        print('X=', X)
        print('y=', y)
        print('nx=', nx)
        print('ny=', ny)
    """
    if not hasattr(model, "use_all_past") or not hasattr(model, "past"):
        raise TypeError(  # pragma: no cover
            "model must be of type BaseTimeSeries not {}".format(type(model)))
    if same_rows:
        if model.use_all_past:
            ncol = X.shape[1] if X is not None else 0
            nrow = y.shape[0] - model.delay2 - model.past + 2

            new_X = numpy.full(
                (y.shape[0], ncol * model.past + model.past), numpy.nan, dtype=y.dtype)
            first = y.shape[0] - nrow
            if X is not None:
                for i in range(0, model.past):
                    begin = i * ncol
                    end = begin + ncol
                    new_X[i:, begin:end] = X[i:]
            for i in range(0, model.past):
                end = y.shape[0] + i + model.delay1 - 1 - model.delay2
                new_X[first - i:first - i + end - i,
                      i + ncol * model.past] = y[i: end]

            new_y = numpy.full(
                (y.shape[0], model.delay2 - model.delay1), numpy.nan, dtype=y.dtype)
            for i in range(model.delay1, model.delay2):
                new_y[first:, i - model.delay1] = y[i + 1:i + nrow + 1]

            new_weights = weights
        else:
            ncol = X.shape[1] if X is not None else 0
            nrow = y.shape[0] - model.delay2 - model.past + 2
            first = y.shape[0] - nrow

            new_X = numpy.full(
                (y.shape[0], ncol + model.past), numpy.nan, dtype=y.dtype)
            if X is not None:
                new_X[first:, :X.shape[1]] = (
                    X[model.past - 1: X.shape[0] - model.delay2 + 1])
            for i in range(model.past):
                end = y.shape[0] + i + model.delay1 - \
                    1 - model.delay2 - model.past + 2
                new_X[first:, i + ncol] = y[i: end]

            new_y = numpy.full(
                (y.shape[0], model.delay2 - model.delay1), numpy.nan, dtype=y.dtype)
            for i in range(model.delay1, model.delay2):
                dec = model.past - 1
                new_y[first:, i - model.delay1] = y[i + dec:i + nrow + dec]
            new_weights = weights
    else:
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
                end = y.shape[0] + i + model.delay1 - \
                    1 - model.delay2 - model.past + 2
                new_X[:, i + ncol] = y[i: end]

            new_y = numpy.empty(
                (nrow, model.delay2 - model.delay1), dtype=y.dtype)
            for i in range(model.delay1, model.delay2):
                dec = model.past - 1
                new_y[:, i - model.delay1] = y[i + dec:i + nrow + dec]
            new_weights = (None if weights is None
                           else weights[model.past - 1:model.past - 1 + nrow])
    return new_X, new_y, new_weights


def check_ts_X_y(model, X, y):
    """
    Checks that datasets *(X, y)* was built with function
    @see fn build_ts_X_y.
    """
    cfg = get_config()
    if cfg.get('assume_finite', True):
        return  # pragma: no cover
    if X.dtype not in (numpy.float32, numpy.float64):
        raise TypeError(
            "Features must be of type float32 and float64 not {}.".format(X.dtype))
    if y is not None and y.dtype not in (numpy.float32, numpy.float64):
        raise TypeError(  # pragma: no cover
            "Features must be of type float32 and float64 not {}.".format(y.dtype))
    cst = model.past
    if (hasattr(model, 'preprocessing_') and model.preprocessing_ is not None):
        cst += model.preprocessing_.context_length
    if y is None:
        if cst > 0:
            raise AssertionError(  # pragma: no cover
                "y must be specified to give the model past data to predict, "
                "it requires at least {} observations.".format(cst))
        return  # pragma: no cover
    if y.shape[0] != X.shape[0]:
        raise AssertionError(  # pragma: no cover
            "X and y must have the same number of rows {} != {}.".format(
                X.shape[0], y.shape[0]))
    if len(y.shape) > 1 and y.shape[1] != 1:
        raise AssertionError(  # pragma: no cover
            "y must be 1-dimensional not has shape {}.".format(y.shape))
    if y.shape[0] < cst:
        raise AssertionError(  # pragma: no cover
            "y is not enough past data to predict, "
            "it requires at least {} observations.".format(cst))
