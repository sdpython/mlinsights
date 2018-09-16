"""
@file
@brief Implements a way to get close examples based
on the output of a machine learned model.
"""
from .search_engine_predictions import SearchEnginePredictions


class SearchEnginePredictionImages(SearchEnginePredictions):
    """
    Extends class @see cl SearchEnginePredictions.
    Vectors are coming from images. The metadata must contains
    information about path names. We assume all images can hold
    in memory. An example can found in notebook
    :ref:`searchimageskerasrst` or :ref:`searchimagestorchrst`.
    """

    def _prepare_fit(self, data=None, features=None, metadata=None, transform=None, n=None, fLOG=None):
        """
        Stores data in the class itself.

        @param      data        a dataframe or None if the
                                the features and the metadata
                                are specified with an array and a
                                dictionary
        @param      features    features columns or an array
        @param      metadata    data
        @param      transform   transform each vector before using it
        @param      n           takes *n* images (or ``len(iter_images)``)
        @param      fLOG        logging function
        """
        iter_images = data
        # We delay the import as keras backend is not necessarily installed.
        from keras.preprocessing.image import Iterator
        if not isinstance(iter_images, Iterator):
            raise NotImplementedError(
                "iter_images must be a keras Iterator. No other option implemented.")
        if iter_images.batch_size != 1:
            raise ValueError("batch_size must be 1 not {0}".format(
                iter_images.batch_size))
        self.iter_images_ = iter_images
        if n is None:
            n = len(iter_images)
        if not hasattr(iter_images, "filenames"):
            raise NotImplementedError(
                "Iterator does not iterate on images but numpy arrays (not implemented).")

        def get_current_index(flow):
            "get current index"
            return flow.index_array[(flow.batch_index + flow.n - 1) % flow.n]

        def iterator_feature_meta():
            "iterators on metadaat"
            for i, im in zip(range(n), iter_images):
                name = iter_images.filenames[get_current_index(iter_images)]
                yield im[0], dict(name=name)
                if fLOG and i % 10000 == 0:
                    fLOG(
                        '[SearchEnginePredictionImages.fit] i={}/{} - {}'.format(i, n, name))

        super()._prepare_fit(data=iterator_feature_meta(), transform=transform)

    def fit(self, iter_images, n=None, fLOG=None):
        """
        Processes images through the model and fits a *k-nn*.

        @param      iter_images `Iterator <https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L719>`_
        @param      n           takes *n* images (or ``len(iter_images)``)
        @param      fLOG        logging function
        @param      kwimg       parameters used to preprocess the images
        """
        self._prepare_fit(data=iter_images, transform=self.fct, n=n, fLOG=fLOG)
        return self._fit_knn()

    def kneighbors(self, iter_images, n_neighbors=None):
        """
        Searches for neighbors close to the first image
        returned by *iter_images*. It returns the neighbors
        only for the first image.

        @param      iter_images `Iterator <https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L719>`_
        @return                 score, ind, meta

        *score* is an array representing the lengths to points,
        *ind* contains the indices of the nearest points in the population matrix,
        *meta* is the metadata.
        """
        # We delay the import as keras backend is not necessarily installed.
        from keras.preprocessing.image import Iterator
        if not isinstance(iter_images, Iterator):
            raise NotImplementedError(
                "iter_images must be a keras Iterator. No other option implemented.")
        if iter_images.batch_size != 1:
            raise ValueError("batch_size must be 1 not {0}".format(
                iter_images.batch_size))
        for img in iter_images:
            X = img[0]
            break
        return super().kneighbors(X, n_neighbors=n_neighbors)
