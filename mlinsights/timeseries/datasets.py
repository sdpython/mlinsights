"""
@file
@brief Datasets for timeseries.
"""
import datetime
import numpy
import pandas


def artificial_data(dt1, dt2, minutes=1):
    """
    Generates articial data every minutes.

    @param      dt1     first date
    @param      dt2     second date
    @param      minutes interval between two observations
    @return             dataframe

    .. runpython::
        :showcode:

        import datetime
        from mlinsights.timeseries.datasets import artificial_data

        now = datetime.datetime.now()
        data = artificial_data(now - datetime.timedelta(40), now)
        print(data.head())
    """

    def fxweek(x):
        return 2 - x * (1 - x)

    def sat(x):
        return 2 * x + 2

    data = []
    dt = datetime.timedelta(minutes=minutes)
    while dt1 < dt2:
        if dt1.weekday() == 6:
            dt1 += dt
            continue
        if minutes <= 120 and not (dt1.hour >= 8 and dt1.hour <= 18):
            dt1 += dt
            continue
        x = (dt1.hour - 8) / 10
        if dt1.weekday() == 5:
            y = sat(x)
        else:
            y = fxweek(x)
        data.append({'time': dt1, 'y': y})
        dt1 += dt
    df = pandas.DataFrame(data)
    df['y'] += numpy.random.randn(df.shape[0]) * 0.1
    df['time'] = pandas.DatetimeIndex(df['time'])
    return df
