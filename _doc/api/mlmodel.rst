=======================
Machine Learning Models
=======================

.. contents::
    :local:

Helpers
=======

model_featurizer
++++++++++++++++

.. autofunction:: mlinsights.mlmodel.ml_featurizer.model_featurizer

Clustering
==========

ConstraintKMeans
++++++++++++++++

.. autoclass:: mlinsights.mlmodel.kmeans_constraint.ConstraintKMeans
    :members:

KMeansL1L2
++++++++++

.. autoclass:: mlinsights.mlmodel.kmeans_l1.KMeansL1L2
    :members:

Trainers
========

ClassifierAfterKMeans
+++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.classification_kmeans.ClassifierAfterKMeans
    :members:

CustomizedMultilayerPerceptron
++++++++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.quantile_mlpregressor.CustomizedMultilayerPerceptron
    :members:

IntervalRegressor
+++++++++++++++++

.. autoclass:: mlinsights.mlmodel.interval_regressor.IntervalRegressor
    :members:

ApproximateNMFPredictor
+++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.anmf_predictor.ApproximateNMFPredictor
    :members:

PiecewiseClassifier
+++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.piecewise_estimator.PiecewiseClassifier
    :members:

PiecewiseRegressor
++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.piecewise_estimator.PiecewiseRegressor
    :members:

PiecewiseTreeRegressor
++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.piecewise_tree_regression.PiecewiseTreeRegressor
    :members:

QuantileMLPRegressor
++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.quantile_mlpregressor.QuantileMLPRegressor
    :members:

QuantileLinearRegression
++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.quantile_regression.QuantileLinearRegression
    :members:

TransformedTargetClassifier2
++++++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.target_predictors.TransformedTargetClassifier2
    :members:

TransformedTargetRegressor2
+++++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.target_predictors.TransformedTargetRegressor2
    :members:

Transforms
==========

NGramsMixin
+++++++++++

.. autoclass:: mlinsights.mlmodel.sklearn_text.NGramsMixin
    :members:

BaseReciprocalTransformer
+++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.sklearn_transform_inv.BaseReciprocalTransformer
    :members:

CategoriesToIntegers
++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.categories_to_integers.CategoriesToIntegers
    :members:

ExtendedFeatures
++++++++++++++++

.. autoclass:: mlinsights.mlmodel.extended_features.ExtendedFeatures
    :members:

FunctionReciprocalTransformer
+++++++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.sklearn_transform_inv_fct.FunctionReciprocalTransformer
    :members:

PermutationReciprocalTransformer
++++++++++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.sklearn_transform_inv_fct.PermutationReciprocalTransformer
    :members:

PredictableTSNE
+++++++++++++++

.. autoclass:: mlinsights.mlmodel.predictable_tsne.PredictableTSNE
    :members:

TransferTransformer
+++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.transfer_transformer.TransferTransformer
    :members:

TraceableCountVectorizer
++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.sklearn_text.TraceableCountVectorizer
    :members:

TraceableTfidfVectorizer
++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.sklearn_text.TraceableTfidfVectorizer
    :members:

Exploration
===========

The following implementation play with :epkg:`scikit-learn`
API, it overwrites the code handling parameters.

SkBaseTransformLearner
++++++++++++++++++++++

.. autoclass:: mlinsights.sklapi.sklearn_base_transform_learner.SkBaseTransformLearner
    :members:

SkBaseTransformStacking
+++++++++++++++++++++++

.. autoclass:: mlinsights.sklapi.sklearn_base_transform_stacking.SkBaseTransformStacking
    :members:

Exploration in C
================

The following classes require :epkg:`scikit-learn` *>= 1.3.0*,
otherwise, they do not get compiled.

SimpleRegressorCriterion
++++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.piecewise_tree_regression_criterion.SimpleRegressorCriterion
    :members:

SimpleRegressorCriterionFast
++++++++++++++++++++++++++++

A similar design but a much faster implementation close to what
:epkg:`scikit-learn` implements.

.. autoclass:: mlinsights.mlmodel.piecewise_tree_regression_criterion_fast.SimpleRegressorCriterionFast
    :members:

LinearRegressorCriterion
++++++++++++++++++++++++

The next one implements a criterion which optimizes the mean square error
assuming the points falling into one node of the tree are approximated by
a line. The mean square error is the error made with a linear regressor
and not a constant anymore. The documentation will be completed later.

`mlinsights.mlmodel.piecewise_tree_regression_criterion_linear.LinearRegressorCriterion`

`mlinsights.mlmodel.piecewise_tree_regression_criterion_linear_fast.SimpleRegressorCriterionFast`

Losses
++++++

.. autofunction:: mlinsights.mlmodel.quantile_mlpregressor.absolute_loss

Hidden API
==========

_switch_clusters
++++++++++++++++

.. autofunction:: mlinsights.mlmodel._kmeans_constraint_._switch_clusters
