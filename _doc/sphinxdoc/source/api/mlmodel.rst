
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

.. autosignature:: mlinsights.mlmodel.piecewise_estimator.PiecewiseClassifier

.. autosignature:: mlinsights.mlmodel.piecewise_estimator.PiecewiseRegression

.. autosignature:: mlinsights.mlmodel.quantile_mlpregressor.QuantileMLPRegressor

.. autosignature:: mlinsights.mlmodel.quantile_regression.QuantileLinearRegression

Transforms
++++++++++

.. autosignature:: mlinsights.mlmodel.categories_to_integers.CategoriesToIntegers

.. autosignature:: mlinsights.mlmodel.extended_features.ExtendedFeatures

.. autosignature:: mlinsights.mlmodel.predictable_tsne.PredictableTSNE

.. autosignature:: mlinsights.mlmodel.transfer_transformer.TransferTransformer

Exploration
+++++++++++

The following implementation play with :epkg:`scikit-learn`
API, it overwrites the code handling parameters.

.. autosignature:: mlinsights.sklapi.sklearn_base_transform_learner.SkBaseTransformLearner

.. autosignature:: mlinsights.sklapi.sklearn_base_transform_stacking.SkBaseTransformStacking
