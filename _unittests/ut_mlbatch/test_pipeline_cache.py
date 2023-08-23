# -*- coding: utf-8 -*-
import unittest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD as SVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlbatch.pipeline_cache import PipelineCache
from mlinsights.mlbatch.cache_model import MLCache
from mlinsights.mlmodel.sklearn_testing import clone_with_fitted_parameters


class TestPipelineCache(ExtTestCase):
    def test_make_classification(self):
        X, y = make_classification(random_state=42)

        pipe0 = Pipeline([("pca", PCA(2)), ("lr", LogisticRegression())])
        pipe = PipelineCache([("pca", PCA(2)), ("lr", LogisticRegression())], "cache__")

        if hasattr(pipe0, "_check_fit_params"):
            pars0 = pipe0._check_fit_params()  # pylint: disable=W0212,E1101
            pars1 = pipe._check_fit_params()  # pylint: disable=W0212,E1101
            self.assertEqual(pars0, pars1)

        pipe0.fit(X, y)
        pipe.fit(X, y)
        cache = MLCache.get_cache("cache__")
        self.assertEqual(len(cache), 1)
        key = list(cache.keys())[0]
        self.assertIn("[('X',", key)
        self.assertIn("('copy', 'True')", key)
        MLCache.remove_cache("cache__")
        items = list(pipe.cache_.items())
        self.assertEqual(len(items), 1)
        self.assertEqual(cache.count("A"), 0)

    def test_pass_through(self):
        X, y = make_classification(random_state=42)
        pipe = Pipeline([("pca", PCA(2)), ("p", "passthrough")])
        pipe.fit(X, y)

    def test_grid_search(self):
        X, y = make_classification(random_state=42)
        param_grid = {
            "pca__n_components": [2, 3],
            "pca__whiten": [True, False],
            "lr__fit_intercept": [True, False],
        }
        pipe = Pipeline([("pca", PCA(2)), ("lr", LogisticRegression())])
        grid0 = GridSearchCV(pipe, param_grid, error_score="raise")
        grid0.fit(X, y)

        pipe = PipelineCache(
            [("pca", PCA(2)), ("lr", LogisticRegression())], "cache__2"
        )
        grid = GridSearchCV(pipe, param_grid, error_score="raise")

        grid.fit(X, y)
        cache = MLCache.get_cache("cache__2")
        # 0.22 increases the number of cached results
        self.assertIn(len(cache), (13, 21))
        key = list(cache.keys())[0]
        self.assertIn("[('X',", key)
        self.assertIn("('copy', 'True')", key)
        MLCache.remove_cache("cache__2")
        self.assertEqual(grid0.best_params_, grid.best_params_)

    def test_grid_search_1(self):
        X, y = make_classification(random_state=42)
        param_grid = {
            "pca__n_components": [2, 3],
            "pca__whiten": [True, False],
            "lr__fit_intercept": [True, False],
        }
        pipe = Pipeline([("pca", PCA(2)), ("lr", LogisticRegression())])
        grid0 = GridSearchCV(pipe, param_grid, error_score="raise", n_jobs=1)
        grid0.fit(X, y)

        pipe = PipelineCache(
            [("pca", PCA(2)), ("lr", LogisticRegression())], "cache__1"
        )
        grid = GridSearchCV(pipe, param_grid, error_score="raise", n_jobs=1)

        grid.fit(X, y)
        cache = MLCache.get_cache("cache__1")
        # 0.22 increases the number of cached results
        self.assertIn(len(cache), (13, 21))
        key = list(cache.keys())[0]
        self.assertIn("[('X',", key)
        self.assertIn("('copy', 'True')", key)
        MLCache.remove_cache("cache__1")
        self.assertEqual(grid0.best_params_, grid.best_params_)

    def test_grid_search_model(self):
        X, y = make_classification(random_state=42)
        param_grid = [
            {"pca": [PCA(2)], "lr__fit_intercept": [False, True]},
            {"pca": [SVD(2)], "lr__fit_intercept": [False, True]},
        ]
        pipe = Pipeline([("pca", "passthrough"), ("lr", LogisticRegression())])
        grid0 = GridSearchCV(pipe, param_grid, error_score="raise")
        grid0.fit(X, y)

        pipe = PipelineCache(
            [("pca", "passthrough"), ("lr", LogisticRegression())], "cache__3"
        )
        grid = GridSearchCV(pipe, param_grid, error_score="raise")

        grid.fit(X, y)
        cache = MLCache.get_cache("cache__3")
        # 0.22 increases the number of cached results
        self.assertIn(len(cache), (7, 11))
        key = list(cache.keys())[0]
        self.assertIn("[('X',", key)
        self.assertIn("('copy', 'True')", key)
        MLCache.remove_cache("cache__3")
        self.assertEqual(grid0.best_params_, grid.best_params_)

    def test_clone_with_fitted_parameters(self):
        X, y = make_classification(random_state=42)
        pipe = Pipeline([("pca", PCA(2)), ("lr", LogisticRegression())])
        pipe.fit(X, y)
        cl = clone_with_fitted_parameters(pipe)
        self.assertNotEmpty(cl)
        cl = clone_with_fitted_parameters([pipe])
        self.assertIsInstance(cl, list)
        cl = clone_with_fitted_parameters((pipe,))
        self.assertIsInstance(cl, tuple)


if __name__ == "__main__":
    unittest.main()
