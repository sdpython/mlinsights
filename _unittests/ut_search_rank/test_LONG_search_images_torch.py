# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr
from io import StringIO
import pandas
import numpy
from mlinsights.ext_test_case import ExtTestCase, unzip_files


class TestSearchPredictionsImagesTorch(ExtTestCase):
    def test_search_predictions_torch(self):
        from mlinsights.search_rank import SearchEnginePredictionImages

        # We delay the import as keras backend is not necessarily available.
        with redirect_stderr(StringIO()):
            try:
                import torchvision.models as tmodels
            except (SyntaxError, ModuleNotFoundError) as e:
                warnings.warn(f"torch is not available: {e}")
                return
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader

        # deep learning model
        model = tmodels.squeezenet1_1(pretrained=True)

        # images
        this = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as temp:
            sub = os.path.join(temp, "simages")
            os.mkdir(sub)
            zipimg = os.path.join(
                this,
                "..",
                "..",
                "_doc",
                "examples",
                "data",
                "dog-cat-pixabay.zip",
            )
            files = unzip_files(zipimg, where_to=sub)
            self.assertTrue(len(files) > 0)

            # sequence of images
            trans = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
            imgs_ = datasets.ImageFolder(temp, trans)
            dataloader = DataLoader(imgs_, batch_size=1, shuffle=False, num_workers=1)
            img_seq = iter(dataloader)
            imgs = list(img[0] for img in img_seq)

            # search
            se = SearchEnginePredictionImages(model, n_neighbors=5)
            r = repr(se)
            self.assertIn("SearchEnginePredictionImages", r)

            # fit
            se.fit(imgs_)

            # neighbors
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
            self.assertEndsWith(
                "simages/cat-1151519__480.jpg", meta.loc[0, "name"].replace("\\", "/")
            )

            # neighbors 2
            score, ind, meta = se.kneighbors(imgs)

            self.assertIsInstance(ind, (list, numpy.ndarray))
            self.assertIsInstance(score, numpy.ndarray)
            self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))


if __name__ == "__main__":
    unittest.main()
