"""
@file
@brief Shortcuts to *mlmodel*.
"""
from .categories_to_integers import CategoriesToIntegers
from .classification_kmeans import ClassifierAfterKMeans
from .extended_features import ExtendedFeatures
from .interval_regressor import IntervalRegressor
from .ml_featurizer import model_featurizer
from .anmf_predictor import ApproximateNMFPredictor
from .piecewise_estimator import PiecewiseRegressor, PiecewiseClassifier
from .predictable_tsne import PredictableTSNE
from .quantile_mlpregressor import QuantileMLPRegressor
from .quantile_regression import QuantileLinearRegression
from .sklearn_testing import test_sklearn_pickle, test_sklearn_clone, test_sklearn_grid_search_cv
from .sklearn_text import TraceableTfidfVectorizer, TraceableCountVectorizer
from .sklearn_transform_inv_fct import FunctionReciprocalTransformer, PermutationReciprocalTransformer
from .target_predictors import TransformedTargetClassifier2, TransformedTargetRegressor2
from .transfer_transformer import TransferTransformer
