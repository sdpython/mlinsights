"""
@file
@brief Implements a way to get close examples based
on the output of a machine learned model.
"""
import numpy
from .search_engine_predictions import SearchEnginePredictions


class SearchEnginePredictionImages(SearchEnginePredictions):
    """
    Extends class @see cl SearchEnginePredictions.
    Vectors are coming from images. The metadata must contains
    information about path names. We assume all images can hold
    in memory. An example can found in notebook
    :ref:`searchimageskerasrst` or :ref:`searchimagestorchrst`.
    Another example can be found there:
    `search_images_dogcat.py
    <https://github.com/sdpython/ensae_projects/blob/master/src/
    ensae_projects/restapi/search_images_dogcat.py>`_.
    """

    def _prepare_fit(self, data=None, features=None, metadata=None,
                     transform=None, n=None, fLOG=None):
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
        if "torch" in str(type(data)):
            self.module_ = "torch"
            from torch.utils.data import DataLoader  # pylint: disable=E0401,C0415,E0611
            dataloader = DataLoader(
                data, batch_size=1, shuffle=False, num_workers=0)
            self.iter_images_ = iter_images = iter(
                zip(dataloader, data.samples))
            if n is None:
                n = len(data)
        elif "keras" in str(type(data)):  # pragma: no cover
            self.module_ = "keras"
            iter_images = data
            # We delay the import as keras backend is not necessarily installed.
            from keras.preprocessing.image import Iterator  # pylint: disable=E0401,C0415,E0611
            from keras_preprocessing.image import DirectoryIterator, NumpyArrayIterator  # pylint: disable=E0401,C0415
            if not isinstance(iter_images, (Iterator, DirectoryIterator, NumpyArrayIterator)):
                raise NotImplementedError(  # pragma: no cover
                    "iter_images must be a keras Iterator. No option implemented for type {0}."
                    "".format(type(iter_images)))
            if iter_images.batch_size != 1:
                raise ValueError(  # pragma: no cover
                    "batch_size must be 1 not {0}".format(
                        iter_images.batch_size))
            self.iter_images_ = iter_images
            if n is None:
                n = len(iter_images)
            if not hasattr(iter_images, "filenames"):
                raise NotImplementedError(  # pragma: no cover
                    "Iterator does not iterate on images but numpy arrays (not implemented).")
        else:
            raise TypeError(  # pragma: no cover
                "Unexpected data type {0}.".format(type(data)))

        def get_current_index(flow):
            "get current index"
            return flow.index_array[(flow.batch_index + flow.n - 1) % flow.n]

        def iterator_feature_meta():
            "iterators on metadata"
            def accessor(iter_images):
                if hasattr(iter_images, 'filenames'):
                    # keras
                    return (lambda i, ite: (ite, iter_images.filenames[get_current_index(iter_images)]))
                else:
                    # torch
                    return (lambda i, ite: (ite[0], ite[1][0]))
            acc = accessor(iter_images)

            for i, it in zip(range(n), iter_images):
                im, name = acc(i, it)
                if not isinstance(name, str):
                    raise TypeError(  # pragma: no cover
                        "name should be a string, not {0}".format(type(name)))
                yield im[0], dict(name=name, i=i)
                if fLOG and i % 10000 == 0:
                    fLOG(
                        '[SearchEnginePredictionImages.fit] i={}/{} - {}'.format(i, n, name))
        super()._prepare_fit(data=iterator_feature_meta(), transform=transform)

    def fit(self, iter_images, n=None, fLOG=None):  # pylint: disable=W0237
        """
        Processes images through the model and fits a *k-nn*.

        @param      iter_images `Iterator <https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L719>`_
        @param      n           takes *n* images (or ``len(iter_images)``)
        @param      fLOG        logging function
        @param      kwimg       parameters used to preprocess the images
        """
        self._prepare_fit(data=iter_images, transform=self.fct, n=n, fLOG=fLOG)
        return self._fit_knn()

    def kneighbors(self, iter_images, n_neighbors=None):  # pylint: disable=W0237
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
        if isinstance(iter_images, numpy.ndarray):
            if self.module_ == "keras":  # pragma: no cover
                raise NotImplementedError("Not yet implemented or Keras.")
            elif self.module_ == "torch":
                from torch import from_numpy  # pylint: disable=E0611,E0401,C0415
                X = from_numpy(iter_images[numpy.newaxis, :, :, :])
                return super().kneighbors(X, n_neighbors=n_neighbors)
            raise RuntimeError(  # pragma: no cover
                "Unknown module '{0}'.".format(self.module_))
        elif "keras" in str(iter_images):  # pragma: no cover
            if self.module_ != "keras":
                raise RuntimeError(  # pragma: no cover
                    "Keras object but {0} was used to train the KNN.".format(self.module_))
            # We delay the import as keras backend is not necessarily installed.
            # keras, it expects an iterator.
            from keras.preprocessing.image import Iterator  # pylint: disable=E0401,C0415,E0611
            from keras_preprocessing.image import DirectoryIterator, NumpyArrayIterator  # pylint: disable=E0401,C0415,E0611
            if not isinstance(iter_images, (Iterator, DirectoryIterator, NumpyArrayIterator)):
                raise NotImplementedError(  # pragma: no cover
                    "iter_images must be a keras Iterator. No option implemented for type {0}.".format(type(iter_images)))
            if iter_images.batch_size != 1:
                raise ValueError(  # pragma: no cover
                    "batch_size must be 1 not {0}".format(
                        iter_images.batch_size))
            for img in iter_images:
                X = img[0]
                break
            return super().kneighbors(X, n_neighbors=n_neighbors)
        elif "torch" in str(type(iter_images)):
            if self.module_ != "torch":
                raise RuntimeError(  # pragma: no cover
                    "Torch object but {0} was used to train the KNN.".format(self.module_))
            # torch: it expects a tensor
            X = iter_images
            return super().kneighbors(X, n_neighbors=n_neighbors)
        elif isinstance(iter_images, list):
            res = [self.kneighbors(it, n_neighbors=n_neighbors)
                   for it in iter_images]
            return (numpy.vstack([_[0] for _ in res]),
                    numpy.vstack([_[1] for _ in res]),
                    numpy.vstack([_[2] for _ in res]))
        else:
            raise TypeError(  # pragma: no cover
                "Unexpected type {0} in SearchEnginePredictionImages.kneighbors".format(
                    type(iter_images)))
