import numpy
from .search_engine_predictions import SearchEnginePredictions


class SearchEnginePredictionImages(SearchEnginePredictions):
    """
    Extends class :class:`SearchEnginePredictions`.
    Vectors are coming from images. The metadata must contains
    information about path names. We assume all images can hold
    in memory. An example can found in notebook
    :ref:`searchimageskerasrst` or :ref:`searchimagestorchrst`.

    Stores data in the class itself.

    :param data: a dataframe or None if the
        the features and the metadata
        are specified with an array and a dictionary
    :param features: features columns or an array
    :param metadata: data
    :param transform: transform each vector before using it
    :param n: takes *n* images (or ``len(iter_images)``)
    :param verbose: verbosity
    """

    def _prepare_fit(
        self, data=None, features=None, metadata=None, transform=None, n=None, verbose=0
    ):
        if "torch" in str(type(data)):
            from torch.utils.data import DataLoader

            self.module_ = "torch"

            dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
            self.iter_images_ = iter_images = iter(zip(dataloader, data.samples))
            self.verbose = verbose
            if n is None:
                n = len(data)
        else:
            raise TypeError(f"Unexpected data type {type(data)}.")

        def get_current_index(flow):
            "get current index"
            return flow.index_array[(flow.batch_index + flow.n - 1) % flow.n]

        def iterator_feature_meta():
            "iterators on metadata"

            def accessor(iter_images):
                if hasattr(iter_images, "filenames"):
                    # keras
                    return lambda i, ite: (
                        ite,
                        iter_images.filenames[get_current_index(iter_images)],
                    )
                else:
                    # torch
                    return lambda i, ite: (ite[0], ite[1][0])

            acc = accessor(iter_images)

            for i, it in zip(range(n), iter_images):
                im, name = acc(i, it)
                if not isinstance(name, str):
                    raise TypeError(f"name should be a string, not {type(name)}")
                yield im[0], dict(name=name, i=i)
                if self.verbose and i % 10000 == 0:
                    print(f"[SearchEnginePredictionImages.fit] i={i}/{n} - {name}")

        super()._prepare_fit(data=iterator_feature_meta(), transform=transform)

    def fit(self, iter_images, n=None):
        """
        Processes images through the model and fits a *k-nn*.

        @param      iter_images `Iterator <https://github.com/fchollet/keras/blob/main/keras/preprocessing/image.py#L719>`_
        @param      n           takes *n* images (or ``len(iter_images)``)
        @param      kwimg       parameters used to preprocess the images
        """
        self._prepare_fit(data=iter_images, transform=self.fct, n=n)
        return self._fit_knn()

    def kneighbors(self, iter_images, n_neighbors=None):
        """
        Searches for neighbors close to the first image
        returned by *iter_images*. It returns the neighbors
        only for the first image.

        @param      iter_images `Iterator <https://github.com/fchollet/keras/blob/main/keras/preprocessing/image.py#L719>`_
        @return                 score, ind, meta

        *score* is an array representing the lengths to points,
        *ind* contains the indices of the nearest points in the population matrix,
        *meta* is the metadata.
        """
        if isinstance(iter_images, numpy.ndarray):
            if self.module_ == "keras":
                raise NotImplementedError("Not yet implemented or Keras.")
            elif self.module_ == "torch":
                from torch import from_numpy

                X = from_numpy(iter_images[numpy.newaxis, :, :, :])
                return super().kneighbors(X, n_neighbors=n_neighbors)
            raise RuntimeError(f"Unknown module '{self.module_}'.")
        elif "keras" in str(iter_images):
            if self.module_ != "keras":
                raise RuntimeError(
                    f"Keras object but {self.module_} was used to train the KNN."
                )
            # We delay the import as keras backend is not necessarily installed.
            # keras, it expects an iterator.
            from keras.preprocessing.image import (
                Iterator,
            )
            from keras_preprocessing.image import (
                DirectoryIterator,
                NumpyArrayIterator,
            )

            if not isinstance(
                iter_images, (Iterator, DirectoryIterator, NumpyArrayIterator)
            ):
                raise NotImplementedError(
                    f"iter_images must be a keras Iterator. "
                    f"No option implemented for type {type(iter_images)}."
                )
            if iter_images.batch_size != 1:
                raise ValueError(f"batch_size must be 1 not {iter_images.batch_size}")
            for img in iter_images:
                X = img[0]
                break
            return super().kneighbors(X, n_neighbors=n_neighbors)
        elif "torch" in str(type(iter_images)):
            if self.module_ != "torch":
                raise RuntimeError(
                    f"Torch object but {self.module_} was used to train the KNN."
                )
            # torch: it expects a tensor
            X = iter_images
            return super().kneighbors(X, n_neighbors=n_neighbors)
        elif isinstance(iter_images, list):
            res = [self.kneighbors(it, n_neighbors=n_neighbors) for it in iter_images]
            return (
                numpy.vstack([_[0] for _ in res]),
                numpy.vstack([_[1] for _ in res]),
                numpy.vstack([_[2] for _ in res]),
            )
        else:
            raise TypeError(
                f"Unexpected type {type(iter_images)} in "
                f"SearchEnginePredictionImages.kneighbors"
            )
