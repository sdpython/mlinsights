"""
@file
@brief Timeseries plots.
"""
import calendar
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # pylint: disable=R0402


def plot_week_timeseries(time, value, normalise=True,
                         label=None, h=0.85, value2=None,
                         label2=None, daynames=None,
                         xfmt="%1.0f", ax=None):
    """
    Shows a timeseries dispatched by days as bars.

    @param      time        dates
    @param      value       values to display as bars.
    @param      normalise   normalise data before showing it
    @param      label       label of the series
    @param      values2     second series to show as a line
    @param      label2      label of the second series
    @param      daynames    names to use for week day names (default is English)
    @param      xfmt        format number of the X axis
    @param      ax          existing axis
    @return                 axis

    .. plot::

        import datetime
        import matplotlib.pyplot as plt
        from mlinsights.timeseries.datasets import artificial_data
        from mlinsights.timeseries.agg import aggregate_timeseries
        from mlinsights.timeseries.plotting import plot_week_timeseries

        dt1 = datetime.datetime(2019, 8, 1)
        dt2 = datetime.datetime(2019, 9, 1)
        data = artificial_data(dt1, dt2, minutes=15)
        print(data.head())

        agg = aggregate_timeseries(data, per='week')
        plot_week_timeseries(
            agg['weektime'], agg['y'], label="y",
            value2=agg['y']/2, label2="y/2", normalise=False)
        plt.show()
    """
    if time.shape[0] != value.shape[0]:
        raise AssertionError("Dimension mismatch")  # pragma: no cover

    def coor(ti):
        days = ti.days
        x = days
        y = ti.seconds
        return x, y

    max_value = value.max()
    if value2 is not None:
        max_value = max(max_value, value2.max())
        value2 = value2 / max_value
    value = value / max_value
    input_maxy = 1.

    if ax is None:
        ax = plt.gca()

    # bars
    delta = None
    maxx, maxy = None, None
    first = True
    for i in range(time.shape[0]):
        ti = time[i]
        if i < time.shape[0] - 1:
            ti1 = time[i + 1]
            delta = (ti1 - ti) if delta is None else min(delta, ti1 - ti)
            if delta == 0:
                raise RuntimeError(  # pragma: no cover
                    "The timeseries contains duplicated time values.")
        else:
            ti1 = ti + delta
        x1, y1 = coor(ti)
        x2, y2 = coor(ti1)
        if y2 < y1:
            x2, y2 = coor(ti + delta)
        y2 = y1 + (y2 - y1) * h
        if first and label:
            ax.plot([x1, x1 + value[i] * 0.8], [y1, y1],
                    'b', alpha=0.5, label=label)
            first = False
        if maxx is None:
            maxx = (x1, x1 + input_maxy)
            maxy = (y1, y2)
        else:
            maxx = (min(x1, maxx[0]),  # pylint: disable=E1136
                    max(x1 + input_maxy, maxx[1]))  # pylint: disable=E1136
            maxy = (min(y1, maxy[0]),  # pylint: disable=E1136
                    max(y2, maxy[1]))  # pylint: disable=E1136

        rect = patches.Rectangle((x1, y1), value[i] * h, y2 - y1,
                                 linewidth=1, edgecolor=None,
                                 facecolor='b', fill=True,
                                 alpha=0.5)

        ax.add_patch(rect)

    # days border
    xticks = []
    if daynames is None:
        daynames = list(calendar.day_name)

    maxx = [(maxx[0] // 7) * 7, maxx[1]]
    new_ymin = maxy[0] - (maxy[1] * 0.025 + maxy[0] * 0.975 - maxy[0])
    for i in range(int(maxx[0]), int(maxx[1] + 0.1)):
        x1i = maxx[0] + input_maxy * i
        x2i = x1i + input_maxy
        xticks.append(x1i)
        ax.plot([x1i, x1i + input_maxy], [new_ymin, new_ymin], 'k', alpha=0.5)
        ax.plot([x1i, x1i + input_maxy], [maxy[1], maxy[1]], 'k', alpha=0.5)
        ax.plot([x1i, x1i], [maxy[0], maxy[1]], 'k', alpha=0.5)
        ax.plot([x2i, x2i], [maxy[0], maxy[1]], 'k', alpha=0.5)
        ax.text(x1i, new_ymin, daynames[i])

    # invert y axis
    ax.invert_yaxis()

    # change y labels
    nby = len(ax.get_yticklabels())
    ys = ax.get_yticks()
    ylabels = []
    for i in range(nby):
        dh = ys[i]
        dt = datetime.timedelta(seconds=dh)
        tx = "%dh%02d" % (dt.seconds // 3600,
                          60 * (dt.seconds / 3600 - dt.seconds // 3600))
        ylabels.append(tx)
    ax.set_yticklabels(ylabels)

    # change x labels
    xs = ax.get_xticks()
    xticks = []
    xlabels = []
    for i in range(0, len(xs) - 1):
        if xs[i] < 0:
            continue
        dx = xs[i] - int(xs[i] / input_maxy) * input_maxy
        xlabels.append(dx if normalise else (dx * max_value))
        xticks.append(xs[i])
        dx = (xs[i] + xs[i + 1]) / 2
        dx = dx - int(dx / input_maxy) * input_maxy
        xlabels.append(dx if normalise else (dx * max_value))
        xticks.append((xs[i] + xs[i + 1]) / 2)
    if len(xticks) < len(xlabels):
        xticks.append(xs[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [xfmt % x for x in xlabels] if xfmt else xlabels)

    ax.tick_params(axis='x', rotation=30)

    # value2
    if value2 is not None:
        value = value2.copy()
        if normalise:
            value = value / max_value

        first = True
        xs = []
        ys = []
        for i in range(time.shape[0]):
            ti = time[i]
            if i < time.shape[0] - 1:
                ti1 = time[i + 1]
            else:
                ti1 = ti + delta
            x1, y1 = coor(ti)
            x2, y2 = coor(ti1)
            if y2 < y1:
                x2, y2 = coor(ti + delta)
            y2 = y1 + (y2 - y1) * h

            x2 = x1 + value[i] * h

            if len(ys) > 0 and y2 < ys[-1]:
                if first and label2 is not None:
                    ax.plot(xs, ys, color='orange', linewidth=2, label=label2)
                    first = False
                else:
                    ax.plot(xs, ys, color='orange', linewidth=2)
                xs, ys = [], []

            xs.append(x2)
            ys.append((y1 + y2) / 2)

        if len(xs) > 0:
            ax.plot(xs, ys, color='orange', linewidth=2)

    return ax
