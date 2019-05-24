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
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from mlinsights.search_rank import SearchEnginePredictions


class TestSearchPredictions(ExtTestCase):

    def test_search_predictions_lr(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

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
        self.maxDiff = None
        r = r.replace("solver='warn',", "").replace("solver='lbfgs',", "")
        r = r.replace("l1_ratio=None,", "")
        exp = ("SearchEnginePredictions(fct=LogisticRegression(C=1.0,class_weight=None,dual=False,"
               "fit_intercept=True,intercept_scaling=1,max_iter=100,multi_class='warn',n_jobs=None,"
               "penalty='l2',random_state=None,solver='warn',tol=0.0001,verbose=0,warm_start=False),"
               "fct_params=None,n_neighbors=5)")
        exp = exp.replace("solver='warn',", "").replace("solver='lbfgs',", "")
        exp = exp.replace("l1_ratio=None,", "")
        self.assertEqual(r.replace("\n", "").replace(" ", ""), exp)

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
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

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
        self.assertEqual(r.replace("\n", "").replace(" ", ""),
                         "SearchEnginePredictions(fct=RandomForestClassifier(bootstrap=True,class_weight=None," +
                         "criterion='gini',max_depth=None,max_features='auto',max_leaf_nodes=None," +
                         "min_impurity_decrease=0.0,min_impurity_split=None," +
                         "min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=10," +
                         "n_jobs=None,oob_score=False," +
                         "random_state=None,verbose=0,warm_start=False),fct_params=None,n_neighbors=5)")

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
        self.assertEqual(r.replace("\n", "").replace(" ", ""),
                         "SearchEnginePredictions(fct=RandomForestClassifier(bootstrap=True," +
                         "class_weight=None,criterion='gini'," +
                         "max_depth=None,max_features='auto',max_leaf_nodes=None," +
                         "min_impurity_decrease=0.0,min_impurity_split=None," +
                         "min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0," +
                         "n_estimators=10,n_jobs=None,oob_score=False," +
                         "random_state=None,verbose=0,warm_start=False),fct_params={'output':True}," +
                         "n_neighbors=5)")

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
