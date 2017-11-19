"""
@file
@brief Implements a way to get close examples based
on the output of a machine learned model.
"""
import numpy
from keras.preprocessing.image import Iterator
from .search_engine_predictions import SearchEnginePredictions


class SearchEnginePredictionImages(SearchEnginePredictions):
    """
    Extends class @see cl SearchEnginePredictions.
    Vectors are coming from images. The metadata must contains
    information about path names. We assume all images can hold
    in memory. An example can found in notebook
    :ref:`searchimagesrst`.
    """

    def fit(self, iter_images, n=None):
        """
        Processes images through the model and fits a *k-nn*.

        @param      iter_images `Iterator <https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L719>`_
        @param      n           takes *n* images (or ``len(iter_images)``)
        @param      kwimg       parameters used to preprocess the images
        """
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
            return (flow.batch_index + flow.n - 1) % flow.n

        X = [(im[0], iter_images.filenames[get_current_index(iter_images)])
             for i, im in zip(range(n), iter_images)]
        meta = numpy.array([_[1] for _ in X])
        X = numpy.stack([_[0] for _ in X])
        return super().fit(features=X, metadata=meta)

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
