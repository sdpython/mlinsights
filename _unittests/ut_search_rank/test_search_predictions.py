# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import pandas
import numpy
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pyquickhelper.pycode import ExtTestCase
from mlinsights.search_rank import SearchEnginePredictions


class TestSearchPredictions(ExtTestCase):

    def test_search_predictions_lr(self):
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        y = iris.target
        clf = LogisticRegression()
        clf.fit(X, y)

        res = []
        for i in range(20):
            h = i * 0.05
            h2 = 1 - i * 0.05
            res.append(dict(ind=i * 5, meta1="m%d" %
                            i, meta2="m%d" % (i + 1), f1=h, f2=h2))
        df = pandas.DataFrame(res)

        se = SearchEnginePredictions(clf, n_neighbors=5)
        r = repr(se)
        exp = "SearchEnginePredictions(fct=LogisticRegression("
        self.assertStartsWith(exp, r)

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

    def test_search_predictions_rfc(self):
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        y = iris.target
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X, y)

        res = []
        for i in range(20):
            h = i * 0.05
            h2 = 1 - i * 0.05
            res.append(dict(ind=i * 5, meta1="m%d" %
                            i, meta2="m%d" % (i + 1), f1=h, f2=h2))
        df = pandas.DataFrame(res)

        # trees output
        se = SearchEnginePredictions(clf, n_neighbors=5)
        r = repr(se)
        rr = r.replace("\n", "").replace(" ", "")
        self.assertIn(
            "SearchEnginePredictions(fct=RandomForestClassifier(", rr)
        self.assertIn("fct_params=None", rr)

        se.fit(data=None, features=df[["f1", "f2"]].values,
               metadata=df[["ind", "meta1", "meta2"]])
        score, ind, meta = se.kneighbors([0.5, 0.5])

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 1)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertEqual(score[0], 0)

        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))
        self.assertEqual(meta.shape, (5, 3))
        self.assertEqual(meta.iloc[0, 0], 5)

        # classifier output
        se = SearchEnginePredictions(
            clf, fct_params={'output': True}, n_neighbors=5)
        r = repr(se)
        rr = r.replace("\n", "").replace(" ", "")
        self.assertIn(
            "SearchEnginePredictions(fct=RandomForestClassifier(", rr)
        self.assertIn("fct_params={'output':True}", rr)

        se.fit(data=None, features=df[["f1", "f2"]].values,
               metadata=df[["ind", "meta1", "meta2"]])
        score, ind, meta = se.kneighbors([0.5, 0.5])

        self.assertIsInstance(ind, (list, numpy.ndarray))
        self.assertEqual(len(ind), 5)
        self.assertEqual(ind[0], 1)

        self.assertIsInstance(score, numpy.ndarray)
        self.assertEqual(score.shape, (5,))
        self.assertEqual(score[0], 0)

        self.assertIsInstance(meta, (numpy.ndarray, pandas.DataFrame))
        self.assertEqual(meta.shape, (5, 3))
        self.assertEqual(meta.iloc[0, 0], 5)


if __name__ == "__main__":
    unittest.main()
