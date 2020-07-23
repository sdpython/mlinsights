"""
@file
@brief Implements a way to get close examples based
on the output of a machine learned model.
"""
import json
import zipfile
import pandas
import numpy
from sklearn.neighbors import NearestNeighbors
from pandas_streaming.df import to_zip, read_zip
from ..helpers.parameters import format_function_call


class SearchEngineVectors:
    """
    Implements a kind of local search engine which
    looks for similar results assuming they are vectors.
    The class is using
    :epkg:`sklearn:neighborsNearestNeighbors` to find
    the nearest neighbors of a vector and follows
    the same API.
    The class populates members:

    * ``features_``: vectors used to compute the neighbors
    * ``knn_``: parameters for the :epkg:`sklearn:neighborsNearestNeighbors`
    * ``metadata_``: metadata, can be None
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

    def _is_iterable(self, data):
        """
        Tells if an objet is an iterator or not.
        """
        try:
            iter(data)
            return not isinstance(data, (list, tuple, pandas.DataFrame, numpy.ndarray))
        except TypeError:
            return False

    def _prepare_fit(self, data=None, features=None, metadata=None, transform=None):
        """
        Stores data in the class itself.

        @param      data        a :epkg:`dataframe` or None if the
                                the features and the metadata
                                are specified with an array and a
                                dictionary
        @param      features    features columns or an array
        @param      metadata    data
        @param      transform   transform each vector before using it

        *transform* is a function whose signature::

            def transform(vec, many):
                # Many tells is the functions receives many vectors
                # or just one (many=False).

        Function *transform* is applied only if
        *data* is not None.
        """
        iterate = self._is_iterable(data)
        if iterate:
            if data is None:
                raise ValueError(  # pragma: no cover
                    "iterator is True, data must be specified.")
            if features is not None:
                raise ValueError(  # pragma: no cover
                    "iterator is True, features must be None.")
            if metadata is not None:
                raise ValueError(  # pragma: no cover
                    "iterator is True, metadata must be None.")
            metas = []
            arrays = []
            for row in data:
                if not isinstance(row, tuple):
                    raise TypeError(  # pragma: no cover
                        'data must be an iterator on tuple')
                if len(row) != 2:
                    raise ValueError(  # pragma: no cover
                        'data must be an iterator on tuple on two elements')
                arr, meta = row
                if not isinstance(meta, dict):
                    raise TypeError(  # pragma: no cover
                        'Second element of the tuple must be a dictionary')
                metas.append(meta)
                if transform is None:
                    tradd = arr
                else:
                    tradd = transform(arr, False)
                if not isinstance(tradd, numpy.ndarray):
                    if transform is None:
                        raise TypeError(  # pragma: no cover
                            "feature should be of type numpy.array not {}".format(type(tradd)))
                    else:
                        raise TypeError(  # pragma: no cover
                            "output of method transform ({}) should be of type numpy.array not {}".format(
                                transform, type(tradd)))
                arrays.append(tradd)
            self.features_ = numpy.vstack(arrays)
            self.metadata_ = pandas.DataFrame(metas)
        elif data is None:
            if not isinstance(features, numpy.ndarray):
                raise TypeError(  # pragma: no cover
                    "features must be an array if data is None")
            self.features_ = features
            self.metadata_ = metadata
        else:
            if not isinstance(data, pandas.DataFrame):
                raise ValueError(  # pragma: no cover
                    "data should be a dataframe")
            self.features_ = data[features]
            self.metadata_ = data[metadata] if metadata else None

    def fit(self, data=None, features=None, metadata=None):
        """
        Every vector comes with a list of metadata.

        @param      data        a dataframe or None if the
                                the features and the metadata
                                are specified with an array and a
                                dictionary
        @param      features    features columns or an array
        @param      metadata    data
        """
        self._prepare_fit(data=data, features=features, metadata=metadata)
        return self._fit_knn()

    def _fit_knn(self):
        """
        Fits the nearest neighbors.
        """
        self.knn_ = NearestNeighbors(**self.pknn)
        self.knn_.fit(self.features_)
        return self

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
                raise TypeError(  # pragma: no cover
                    "X must be a list or a vector (1)")
            X = [X]
        if isinstance(X, numpy.ndarray) and (len(X.shape) > 1 and X.shape[0] != 1):
            raise TypeError(  # pragma: no cover
                "X must be a list or a vector (2)")
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
        rind = ind
        if self.metadata_ is None:
            rmeta = None
        elif hasattr(self.metadata_, 'iloc'):
            rmeta = self.metadata_.iloc[ind, :]
        elif len(self.metadata_.shape) == 1:
            rmeta = self.metadata_[ind]
        else:
            rmeta = self.metadata_[ind, :]
        return score, rind, rmeta

    def to_zip(self, zipfilename, **kwargs):
        """
        Saves the features and the metadata into a zipfile.
        The function does not save the *k-nn*.

        @param      zipfilename a :epkg:`*py:zipfile:ZipFile` or a filename
        @param      kwargs      parameters for :epkg:`pandas:to_csv` (for the metadata)
        @return                 zipfilename

        The function relies on function
        `to_zip <http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/pandas_streaming/df/
        dataframe_io.html#pandas_streaming.df.dataframe_io.to_zip>`_.
        It only works for :epkg:`Python` 3.6+.
        """
        if isinstance(zipfilename, str):
            zf = zipfile.ZipFile(zipfilename, 'w')
            close = True
        else:
            zf = zipfilename
            close = False
        if 'index' is not kwargs:
            kwargs['index'] = False
        to_zip(self.features_, zf, 'SearchEngineVectors-features.npy')
        to_zip(self.metadata_, zf, 'SearchEngineVectors-metadata.csv', **kwargs)
        js = json.dumps(self.pknn)
        zf.writestr('SearchEngineVectors-knn.json', js)
        if close:
            zf.close()

    @staticmethod
    def read_zip(zipfilename, **kwargs):
        """
        Restore the features, the metadata to a @see cl SearchEngineVectors.

        @param      zipfilename a :epkg:`*py:zipfile:ZipFile` or a filename
        @param      zname       a filename in th zipfile
        @param      kwargs      parameters for :epkg:`pandas:read_csv`
        @return                 @see cl SearchEngineVectors

        It only works for :epkg:`Python` 3.6+.
        """
        if isinstance(zipfilename, str):
            zf = zipfile.ZipFile(zipfilename, 'r')
            close = True
        else:
            zf = zipfilename
            close = False
        feat = read_zip(zf, 'SearchEngineVectors-features.npy')
        meta = read_zip(zf, 'SearchEngineVectors-metadata.csv', **kwargs)
        js = zf.read('SearchEngineVectors-knn.json')
        knn = json.loads(js)
        if close:
            zf.close()

        obj = SearchEngineVectors(**knn)
        obj.fit(features=feat, metadata=meta)
        return obj
