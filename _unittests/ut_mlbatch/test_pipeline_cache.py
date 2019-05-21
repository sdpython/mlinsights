# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlbatch.pipeline_cache import PipelineCache
from mlinsights.mlbatch.cache_model import MLCache


class TestPipelineCache(ExtTestCase):

    def test_make_classification(self):
        X, y = make_classification(random_state=42)
        pipe = PipelineCache([('pca', PCA(2)),
                              ('lr', LogisticRegression())],
                             'cache__')
        pipe.fit(X, y)
        cache = MLCache.get_cache('cache__')
        self.assertEqual(len(cache), 1)
        key = list(cache.keys())[0]
        self.assertIn("[('X',", key)
        self.assertIn("('copy', 'True')", key)
        MLCache.remove_cache('cache__')


if __name__ == "__main__":
    unittest.main()
