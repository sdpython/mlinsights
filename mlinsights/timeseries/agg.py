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
    if df is None:
        if len(values.shape) == 1:
            df = pandas.DataFrame(dict(time=index, y=values))
            values = 'y'
        else:
            df = pandas.DataFrame(dict(time=index))
            for i in range(values.shape[1]):
                df['y%d' % i] = values[:, i]
            values = list(df.columns)[1:]
        index = 'time'

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
        raise ValueError(  # pragma: no cover
            "Unknown frequency '{}'.".format(per))

    agg_name = _get_column_name(df)
    df = df.copy()
    if unit == 'half-hour':
        freq = datetime.timedelta(minutes=30)
        df[agg_name] = round_(df[index], freq, per)
    else:
        raise ValueError(  # pragma: no cover
            "Unknown time unit '{}'.".format(unit))
    if not isinstance(values, list):
        values = [values]
    if agg == 'sum':
        gr = df[[agg_name] + values].groupby(agg_name, as_index=False).sum()
        agg_name = _get_column_name(gr, 'week' + index)
        gr.columns = [agg_name] + list(gr.columns[1:])
    elif agg == 'norm':
        gr = df[[agg_name] + values].groupby(agg_name, as_index=False).sum()
        agg_name = _get_column_name(gr, 'week' + index)
        agg_cols = list(gr.columns[1:])
        gr.columns = [agg_name] + agg_cols
        for c in agg_cols:
            su = gr[c].sum()
            if su != 0:
                gr[c] /= su
    else:
        raise ValueError(  # pragma: no cover
            "Unknown aggregation '{}'.".format(agg))
    return gr.sort_values(agg_name).reset_index(drop=True)
