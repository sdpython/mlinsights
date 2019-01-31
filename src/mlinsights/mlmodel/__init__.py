"""
@file
@brief Shortcuts to *mlmodel*.
"""
from .predictable_tsne import PredictableTSNE
from .categories_to_integers import CategoriesToIntegers
from .classification_kmeans import ClassifierAfterKMeans
from .quantile_mlpregressor import QuantileMLPRegressor
from .quantile_regression import QuantileLinearRegression
from .sklearn_testing import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv
