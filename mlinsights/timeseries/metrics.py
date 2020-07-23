"""
@file
@brief Timeseries metrics.
"""
import numpy


def ts_mape(expected_y, predicted_y, sample_weight=None):
    """
    Computes :math:`\\frac{\\sum_i | \\hat{Y_t} - Y_t |}
    {\\sum_i | Y_t - Y_{t-1} |}`.
    It compares the prediction to what a dummy
    predictor would do by using the previous day
    as a prediction.

    @param      expected_y          expected values
    @param      predicted_y         predictions
    @return                         metrics
    """
    if len(expected_y) != len(predicted_y):
        raise ValueError(  # pragma: no cover
            'Size mismatch {} != {}.'.format(
                len(expected_y), len(predicted_y)))
    expected_y = numpy.squeeze(expected_y)
    predicted_y = numpy.squeeze(predicted_y)
    mask = numpy.isnan(predicted_y)
    mask2 = mask.copy()
    mask2[1:] |= numpy.isnan(predicted_y[:-1])
    expected_y = numpy.ma.masked_array(expected_y, mask=mask)
    predicted_y = numpy.ma.masked_array(predicted_y, mask=mask2)
    if sample_weight is None:
        dy1 = numpy.sum(numpy.abs(expected_y[:-1] - expected_y[1:]))
        dy2 = numpy.sum(numpy.abs(predicted_y[1:] - expected_y[1:]))
    else:
        dy1 = numpy.sum(
            (numpy.abs(expected_y[:-1] - expected_y[1:]) * sample_weight[1:]))
        dy2 = numpy.sum(
            (numpy.abs(predicted_y[1:] - expected_y[1:]) * sample_weight[1:]))
    dy1 = dy1.sum()
    dy2 = dy2.sum()
    if dy1 == 0:
        return 0 if dy2 == 0 else numpy.infty
    return dy2 / dy1
