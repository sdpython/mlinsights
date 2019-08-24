
Machine Learning Models
=======================

.. contents::
    :local:

Helpers
+++++++

.. autosignature:: mlinsights.mlmodel.ml_featurizer.model_featurizer

Trainers
++++++++

.. autosignature:: mlinsights.mlmodel.classification_kmeans.ClassifierAfterKMeans

.. autosignature:: mlinsights.mlmodel.interval_regressor.IntervalRegressor

.. autosignature:: mlinsights.mlmodel.anmf_predictor.ApproximateNMFPredictor

.. autosignature:: mlinsights.mlmodel.piecewise_estimator.PiecewiseClassifier

.. autosignature:: mlinsights.mlmodel.piecewise_estimator.PiecewiseRegressor

.. autosignature:: mlinsights.mlmodel.piecewise_tree_regression.PiecewiseTreeRegressor

.. autosignature:: mlinsights.mlmodel.quantile_mlpregressor.QuantileMLPRegressor

.. autosignature:: mlinsights.mlmodel.quantile_regression.QuantileLinearRegression

.. autosignature:: mlinsights.mlmodel.target_predictors.TransformedTargetClassifier2

.. autosignature:: mlinsights.mlmodel.target_predictors.TransformedTargetRegressor2

Transforms
++++++++++

.. autosignature:: mlinsights.mlmodel.categories_to_integers.CategoriesToIntegers

.. autosignature:: mlinsights.mlmodel.extended_features.ExtendedFeatures

.. autosignature:: mlinsights.mlmodel.sklearn_transform_inv_fct.FunctionReciprocalTransformer

.. autosignature:: mlinsights.mlmodel.sklearn_transform_inv_fct.PermutationReciprocalTransformer

.. autosignature:: mlinsights.mlmodel.predictable_tsne.PredictableTSNE

.. autosignature:: mlinsights.mlmodel.transfer_transformer.TransferTransformer

.. autosignature:: mlinsights.mlmodel.sklearn_text.TraceableCountVectorizer

.. autosignature:: mlinsights.mlmodel.sklearn_text.TraceableTfidfVectorizer

Exploration
+++++++++++

The following implementation play with :epkg:`scikit-learn`
API, it overwrites the code handling parameters.

.. autosignature:: mlinsights.sklapi.sklearn_base_transform_learner.SkBaseTransformLearner

.. autosignature:: mlinsights.sklapi.sklearn_base_transform_stacking.SkBaseTransformStacking

Exploration in C
++++++++++++++++

The following classes require :epkg:`scikit-learn` *>= 0.21*,
otherwise, they do not get compiled.

.. autosignature:: mlinsights.mlmodel.piecewise_tree_regression_criterion.SimpleRegressorCriterion

A similar design but a much faster implementation close to what
:epkg:`scikit-learn` implements.

.. autosignature:: mlinsights.mlmodel.piecewise_tree_regression_criterion_fast.SimpleRegressorCriterionFast

The next one implements a criterion which optimizes the mean square error
assuming the points falling into one node of the tree are approximated by
a line. The mean square error is the error made with a linear regressor
and not a constant anymore.

.. autosignature:: mlinsights.mlmodel.piecewise_tree_regression_criterion_linear.LinearRegressorCriterion
