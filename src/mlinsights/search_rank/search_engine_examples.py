"""
@file
@brief Implements a way to get close examples based
on the output of a machine learned model.
"""
from ..helpers.parameters import format_function_call
from sklearn.neighbors import NearestNeighbors
import pandas
import numpy


class SearchEngineExamples:
    """
    Implements a kind of local search engine which
    looks for similar results based on the output
    of a function such as the predictions of the machine
    leanrned model. The class is using
    :epkg:`sklearn:neighborsNearestNeighbors` to find
    the nearest neighbors of an example.
    """

    def __init__(self, **pknn):
        """
        @param      pknn        list of parameters, see :epkg:`sklearn:neighborsNearestNeighbors`
        """
        self.pknn = pknn

    def __repr__(self):
        """
        usual
        """
        return format_function_call(self.__class__.__name__, self.pknn)

    def fit(self, data=None, features=None, metadata=None):
        """
        Every observation is described with an id (or index),
        a list of features, a list of metadata.

        @param      data        a dataframe or None if the
                                the features and the metadata
                                are specified with an array and a
                                dictionary
        @param      features    features columns  or
                                or an array
        @param      metadata    data
        """
        if data is None:
            if not isinstance(features, numpy.ndarray):
                raise TypeError("features must be an array if data is None")
            self.features = features
            self.metadata = metadata
            self.index = list(range(features.shape[0]))
        else:
            if not isinstance(data, pandas.DataFrame):
                raise ValueError("data should be a dataframe")
            self.index = list(range(data.shape[0]))
            self.features = data[features]
            self.metadata = data[metadata] if metadata else None

        self.knn_ = NearestNeighbors(**self.pknn)
        self.knn_.fit(self.features)

    def _first_pass(self, X, n_neighbors=None):
        """
        Finds the closest *n_neighbors*.

        @param      X               features
        @param      n_neighbors     number of neighbors to get (default is the value passed to the constructor)
        @return                     *dist*, *ind*

        *dist* is an array representing the lengths to points,
        *ind* contains the indices of the nearest points in the population matrix.
        """
        if isinstance(X, list):
            if len(X) == 0 or isinstance(X[0], (list, tuple)):
                raise TypeError("X must be a list or a vector (1)")
            X = [X]
        if isinstance(X, numpy.ndarray) and (len(X.shape) > 1 and X.shape[0] != 1):
            raise TypeError("X must be a list or a vector (2)")
        dist, ind = self.knn_.kneighbors(
            X, n_neighbors=n_neighbors, return_distance=True)
        ind = ind.ravel()
        dist = dist.ravel()
        return dist, ind

    def _second_pass(self, X, dist, ind):
        """
        Reorders the closest *n_neighbors*.

        @param      X               features
        @param      dist            array representing the lengths to points
        @param      ind             indices of the nearest points in the population matrix
        @return                     *score*, *ind*

        *score* is an array representing the lengths to points,
        *ind* contains the indices of the nearest points in the population matrix.
        """
        return dist, ind

    def kneighbors(self, X, n_neighbors=None):
        """
        Searches for neighbors close to *X*.

        @param      X               features
        @return                     score, ind, meta

        *score* is an array representing the lengths to points,
        *ind* contains the indices of the nearest points in the population matrix,
        *meta* is the metadata
        """
        dist, ind = self._first_pass(X, n_neighbors=n_neighbors)
        score, ind = self._second_pass(X, dist, ind)
        rind = [self.index[i] for i in ind]
        rmeta = self.metadata.iloc[ind,
                                   :] if self.metadata is not None else None
        return score, rind, rmeta
