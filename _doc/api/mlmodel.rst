==================
mlinsights.mlmodel
==================

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

Exploration in C
================

Losses
++++++

.. autofunction:: mlinsights.mlmodel.quantile_mlpregressor.absolute_loss

Hidden API
==========

_switch_clusters
++++++++++++++++

.. autofunction:: mlinsights.mlmodel._kmeans_constraint_._switch_clusters
