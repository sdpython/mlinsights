# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""

import sys
import os
import unittest
import warnings
from contextlib import redirect_stderr
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


class TestSearchPredictionsImagesTorch(ExtTestCase):

    def test_search_predictions_torch(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        from src.mlinsights.search_rank import SearchEnginePredictionImages

        # We delay the import as keras backend is not necessarily available.
        with redirect_stderr(StringIO()):
            try:
                import torchvision.models as tmodels
            except (SyntaxError, ModuleNotFoundError) as e:
                warnings.warn(
                    "torch is not available: {0}".format(e))
                return
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader

        # deep learning model
        model = tmodels.squeezenet1_0(pretrained=True)

        # images
        temp = get_temp_folder(__file__, "temp_search_predictions_torch")
        dest = os.path.join(temp, "simages")
        os.mkdir(dest)
        zipname = os.path.join(
            temp, "..", "..", "..", "_doc", "notebooks", "data", "dog-cat-pixabay.zip")
        files = unzip_files(zipname, where_to=dest)
        self.assertTrue(len(files) > 0)

        # sequence of images
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
        imgs_ = datasets.ImageFolder(temp, trans)
        dataloader = DataLoader(imgs_, batch_size=1,
                                shuffle=False, num_workers=1)
        img_seq = iter(dataloader)
        imgs = list(img[0] for img in img_seq)

        # search
        se = SearchEnginePredictionImages(model, n_neighbors=5)
        r = repr(se)
        self.assertIn("SearchEnginePredictionImages", r)

        # fit
        fLOG('[fit]')
        se.fit(imgs_, fLOG=fLOG)

        # neighbors
        fLOG('[test]')
        score, ind, meta = se.kneighbors(imgs[0])

        # assert
        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 0)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertLess(score[0], 50)

        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))
        self.assertEqual(meta.shape, (5, 2))
        self.assertEndsWith('simages/cat-1151519__480.jpg',
                            meta.iloc[0, 1].replace('\\', '/'))

        # neighbors 2
        score, ind, meta = se.kneighbors(imgs)

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertIsInstance(score, numpy.ndarray)
        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))


if __name__ == "__main__":
    unittest.main()
