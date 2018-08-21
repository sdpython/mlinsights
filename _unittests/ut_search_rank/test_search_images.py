# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""

import sys
import os
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import pandas
import numpy
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.filehelper import unzip_files


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src


class TestSearchPredictionsImages(ExtTestCase):

    def test_search_predictions_keras(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        from src.mlinsights.search_rank import SearchEnginePredictionImages

        # We delay the import as keras backend is not necessarily available.
        with redirect_stderr(StringIO()):
            try:
                from keras.applications.mobilenet import MobileNet
            except SyntaxError as e:
                warnings.warn("tensorflow is probably not available yet on python 3.7: {0}".format(e))
                return
            from keras.preprocessing.image import ImageDataGenerator
            from keras.preprocessing.image import img_to_array, load_img

        # deep learning model
        model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
                          weights='imagenet', input_tensor=None, pooling=None, classes=1000)
        self.assertEqual(model.name, 'mobilenet_1.00_224')

        # images
        temp = get_temp_folder(__file__, "temp_search_predictions_keras")
        dest = os.path.join(temp, "simages")
        os.mkdir(dest)
        zipname = os.path.join(
            temp, "..", "..", "..", "_doc", "notebooks", "data", "dog-cat-pixabay.zip")
        files = unzip_files(zipname, where_to=dest)
        self.assertTrue(len(files) > 0)

        # iterator
        gen = ImageDataGenerator(rescale=1. / 255)
        with redirect_stdout(StringIO()):
            iterim = gen.flow_from_directory(temp, batch_size=1, target_size=(
                224, 224), classes=['simages'], shuffle=False)

        # search
        se = SearchEnginePredictionImages(model, fct_params=dict(
            layer=len(model.layers) - 4), n_neighbors=5)
        r = repr(se)
        self.assertIn("SearchEnginePredictionImages", r)

        # fit
        se.fit(iterim, fLOG=fLOG)

        # neighbors
        score, ind, meta = se.kneighbors(iterim)

        # assert
        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 0)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertEqual(score[0], 0)

        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))
        self.assertEqual(meta.shape, (5, 1))
        self.assertEqual(meta.iloc[0, 0].replace('\\', '/'),
                         'simages/cat-1151519__480.jpg')

        # neighbors 2
        img = load_img(os.path.join(temp, 'simages', 'cat-2603300__480.jpg'),
                       target_size=(224, 224))
        x = img_to_array(img)
        gen = ImageDataGenerator(rescale=1. / 255)
        iterim = gen.flow(x[numpy.newaxis, :, :, :], batch_size=1)
        score, ind, meta = se.kneighbors(iterim)

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertIsInstance(score, numpy.ndarray)
        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))


if __name__ == "__main__":
    unittest.main()
