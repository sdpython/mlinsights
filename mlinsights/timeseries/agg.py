"""
@file
@brief Data aggregation for timeseries.
"""
import datetime
import pandas
from pandas.tseries.frequencies import to_offset


def _get_column_name(df, name='agg'):
    """
    Returns a unique column name not in the existing dataframe.

    @param      df      dataframe
    @param      name    prefix
    @return             new column name
    """
    while name in df.columns:
        name += '_'
    return name


def aggregate_timeseries(df, index='time', values='y',
                         unit='half-hour', agg='sum',
                         per=None):
    """
    Aggregates timeseries assuming the data is in a dataframe.

    @param      df      dataframe
    @param      index   time column
    @param      values  value or values column
    @param      unit    aggregate over a specific period
    @param      sum     kind of aggregation
    @param      per     second aggregation, per week...
    @return             aggregated values
    """

    def round_(serie, freq, per):
        fr = to_offset(freq)
        res = pandas.DatetimeIndex(serie).floor(fr)  # pylint: disable=E1101
        if per is None:
            return res
        if per == 'week':
            pyres = res.to_pydatetime()
            return pandas.to_timedelta(
                map(
                    lambda t: datetime.timedelta(
                        days=t.weekday(), hours=t.hour, minutes=t.minute),
                    pyres))
        if per == 'month':
            pyres = res.to_pydatetime()
            return pandas.to_timedelta(
                map(
                    lambda t: datetime.timedelta(
                        days=t.day, hours=t.hour, minutes=t.minute),
                    pyres))
        raise ValueError("Unknown frequency '{}'.".format(per))

    agg_name = _get_column_name(df)
    df = df.copy()
    if unit == 'half-hour':
        freq = datetime.timedelta(minutes=30)
        df[agg_name] = round_(df[index], freq, per)
    else:
        raise ValueError("Unknown time unit '{}'.".format(unit))
    if not isinstance(values, list):
        values = [values]
    if agg == 'sum':
        gr = df[[agg_name] + values].groupby(agg_name, as_index=False).sum()
        agg_name = _get_column_name(gr, 'week' + index)
        gr.columns = [agg_name, gr.columns[1]]
    else:
        raise ValueError("Unknown aggregation '{}'.".format(agg))
    return gr.sort_values(agg_name).reset_index(drop=True)
