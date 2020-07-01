"""
@file
@brief Find patterns in timeseries.
"""
import numpy
import pandas
from sklearn.cluster import KMeans
from .agg import aggregate_timeseries


def find_ts_group_pattern(ttime, values, names, name_subset=None,
                          per='week', unit='half-hour', agg='sum',
                          estimator=None, fLOG=None):
    """
    Clusters times series to find similar patterns.

    @param      ttime           time column
    @param      values          features to use to cluster
    @param      names           column which holds group name
    @param      name_subset     subset of groups to study, None for all
    @param      per             aggragation per week
    @param      estimator       estimator used to find pattern,
                                :epkg:`sklearn:cluster:KMeans` and
                                10 groups
    @param      fLOG            logging function
    @return                     found clusters, distances
    """
    for var, na in zip([ttime, values, names], ['ttime', 'values', 'names']):
        if not isinstance(var, numpy.ndarray):
            raise TypeError("'{}' must an array not {}".format(na, type(var)))
    # builds features
    set_names = set(names)
    if name_subset is not None:
        set_names &= set(name_subset)
    if fLOG:
        fLOG(  # pragma: no cover
            '[find_ts_group_pattern] build features, {} groups'.format(len(set_names)))
    gr_names = []
    to_merge = []
    for name in set_names:
        indices = names == name
        gr_ttime = ttime[indices]
        gr_values = values[indices]
        gr = aggregate_timeseries(None, gr_ttime, gr_values,
                                  unit=unit, agg=agg, per=per)
        gr.set_index(gr.columns[0], inplace=True)
        gr_names.append(name)
        to_merge.append(gr)

    if fLOG:
        fLOG(  # pragma: no cover
            '[find_ts_group_pattern] merge features')
    all_merged = pandas.concat(to_merge, axis=1)
    all_merged.fillna(0, inplace=True)
    ncol = all_merged.shape[1] // len(gr_names)
    gr_feats = []
    for i, name in enumerate(gr_names):
        feats = all_merged.iloc[:, i * ncol: (i + 1) * ncol].values.ravel()
        gr_feats.append(feats)

    gr_feats = numpy.vstack(gr_feats)

    # cluster
    if fLOG:
        fLOG(  # pragma: no cover
            '[find_ts_group_pattern] clustering, shape={}'.format(gr_feats.shape))
    if estimator is None:
        estimator = KMeans()
    estimator.fit(gr_feats)

    # predicted clusters
    pred = estimator.predict(gr_feats)
    dist = estimator.transform(gr_feats)
    if fLOG:
        fLOG(  # pragma: no cover
            '[find_ts_group_pattern] number of clusters: {}'.format(len(set(pred))))

    row_name = {n: i for i, n in enumerate(gr_names)}
    clusters = numpy.empty(ttime.shape[0], dtype=pred.dtype)
    dists = numpy.empty((ttime.shape[0], dist.shape[1]), dtype=dist.dtype)

    for i in range(ttime.shape[0]):
        if names[i] in row_name:
            index = row_name[names[i]]
            clusters[i] = pred[index]
            dists[i, :] = dist[index, :]
        else:
            clusters[i] = -1
            dists[i, :] = numpy.nan

    return clusters, dists
