
.. image:: https://github.com/sdpython/mlinsights/raw/main/_doc/_static/project_ico.png
    :target: https://github.com/sdpython/mlinsights/

mlinsights: extensions to scikit-learn
======================================

.. image:: https://dev.azure.com/xavierdupre3/mlinsights/_apis/build/status%2Fsdpython.mlinsights%20(2)?branchName=main
    :target: https://dev.azure.com/xavierdupre3/mlinsights/_build/latest?definitionId=16&branchName=main

.. image:: https://badge.fury.io/py/mlinsights.svg
    :target: http://badge.fury.io/py/mlinsights

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://codecov.io/github/sdpython/mlinsights/coverage.svg?branch=main
    :target: https://codecov.io/github/sdpython/mlinsights?branch=main

.. image:: http://img.shields.io/github/issues/sdpython/mlinsights.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/mlinsights/issues

.. image:: https://pepy.tech/badge/mlinsights/month
    :target: https://pepy.tech/project/mlinsights/month
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/mlinsights.svg
    :target: https://github.com/sdpython/mlinsights/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/mlinsights.svg
    :target: https://github.com/sdpython/mlinsights/
    :alt: Stars

.. image:: https://img.shields.io/github/repo-size/sdpython/mlinsights
    :target: https://github.com/sdpython/mlinsights/
    :alt: size

*mlinsights* extends *scikit-learn* with a couple of new models,
transformers, metrics, plotting. It provides new trainers such as
**QuantileLinearRegression** which trains a linear regression with *L1* norm
non-linear correlation based on decision trees, or
**QuantileMLPRegressor** a modification of scikit-learn's MLPRegressor
which trains a multi-layer perceptron with *L1* norm.
It also explores **PredictableTSNE** which trains a supervized
model to replicate *t-SNE* results or a **PiecewiseRegression**
which partitions the data before fitting a model on each bucket.
**PiecewiseTreeRegressor** trains a piecewise regressor using
a linear regression on each piece. **IntervalRegressor** produces
confidence interval by using bootstrapping. **ApproximateNMFPredictor**
approximates a NMF to produce prediction without retraining.

`mlinsights documentation <https://sdpython.github.io/doc/mlinsights/dev/>`_
