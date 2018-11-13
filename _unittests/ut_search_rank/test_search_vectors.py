# -*- coding: utf-8 -*-
"""
@brief      test log(time=1s)
"""

import sys
import os
import unittest
import pandas
import numpy
from sklearn.linear_model import LogisticRegression
from pyquickhelper.pycode import ExtTestCase, get_temp_folder

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

from src.mlinsights.search_rank import SearchEngineVectors


class TestSearchVectors(ExtTestCase):

    def test_import(self):
        self.assertTrue(LogisticRegression is not None)

    def test_search_vectors(self):
        res = []
        for i in range(20):
            h = i * 0.05
            h2 = 1 - i * 0.05
            res.append(dict(ind=i * 5, meta1="m%d" %
                            i, meta2="m%d" % (i + 1), f1=h, f2=h2))
        df = pandas.DataFrame(res)

        se = SearchEngineVectors(n_neighbors=5)
        r = repr(se)
        self.assertEqual(r.replace("\n", "").replace(" ", ""),
                         'SearchEngineVectors(n_neighbors=5)')

        se.fit(data=None, features=df[["f1", "f2"]].values,
               metadata=df[["ind", "meta1", "meta2"]])
        score, ind, meta = se.kneighbors([0.5, 0.5])

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 10)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertEqual(score[0], 0)

        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))
        self.assertEqual(meta.shape, (5, 3))
        self.assertEqual(meta.iloc[0, 0], 50)

        se.fit(data=df, features=["f1", "f2"],
               metadata=["ind", "meta1", "meta2"])
        score, ind, meta = se.kneighbors([0.5, 0.5])

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 10)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertEqual(score[0], 0)

        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))
        self.assertEqual(meta.shape, (5, 3))
        self.assertEqual(meta.iloc[0, 0], 50)

        se.fit(data=df, features=["f1", "f2"])
        score, ind, meta = se.kneighbors([0.5, 0.5])

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 10)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertEqual(score[0], 0)
        self.assertTrue(meta is None)

    def test_search_vectors_zip(self):
        temp = get_temp_folder(__file__, "temp_search_vectors_zip")

        res = []
        for i in range(20):
            h = i * 0.05
            h2 = 1 - i * 0.05
            res.append(dict(ind=i * 5, meta1="m%d" %
                            i, meta2="m%d" % (i + 1), f1=h, f2=h2))
        df = pandas.DataFrame(res)

        se = SearchEngineVectors(n_neighbors=5)
        r = repr(se)
        self.assertEqual(r.replace("\n", "").replace(" ", ""),
                         'SearchEngineVectors(n_neighbors=5)')

        se.fit(data=None, features=df[["f1", "f2"]].values,
               metadata=df[["ind", "meta1", "meta2"]])
        score, ind, meta = se.kneighbors([0.5, 0.5])

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 10)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertEqual(score[0], 0)

        dest = os.path.join(temp, "se.zip")
        se.to_zip(dest, encoding='utf-8')
        se2 = SearchEngineVectors.read_zip(dest, encoding='utf-8')
        score2, ind2, meta2 = se2.kneighbors([0.5, 0.5])
        self.assertEqualArray(score, score2)
        self.assertEqualArray(ind, ind2)
        self.assertEqualDataFrame(meta, meta2)
        self.assertEqual(se.pknn, se2.pknn)


if __name__ == "__main__":
    unittest.main()
