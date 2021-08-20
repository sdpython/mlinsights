
.. image:: https://github.com/sdpython/mlinsights/blob/master/_doc/sphinxdoc/source/phdoc_static/project_ico.png?raw=true
    :target: https://github.com/sdpython/mlinsights/

.. _l-README:

mlinsights - extensions to scikit-learn
=======================================

.. image:: https://travis-ci.com/sdpython/mlinsights.svg?branch=master
    :target: https://app.travis-ci.com/github/sdpython/mlinsights/
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/uj6tq445k3na7hs9?svg=true
    :target: https://ci.appveyor.com/project/sdpython/mlinsights
    :alt: Build Status Windows

.. image:: https://circleci.com/gh/sdpython/mlinsights/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/mlinsights/tree/master

.. image:: https://dev.azure.com/xavierdupre3/mlinsights/_apis/build/status/sdpython.mlinsights%20(2)
    :target: https://dev.azure.com/xavierdupre3/mlinsights/

.. image:: https://badge.fury.io/py/mlinsights.svg
    :target: http://badge.fury.io/py/mlinsights

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://codecov.io/github/sdpython/mlinsights/coverage.svg?branch=master
    :target: https://codecov.io/github/sdpython/mlinsights?branch=master

.. image:: http://img.shields.io/github/issues/sdpython/mlinsights.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/mlinsights/issues

.. image:: http://www.xavierdupre.fr/app/mlinsights/helpsphinx/_images/nbcov.png
    :target: http://www.xavierdupre.fr/app/mlinsights/helpsphinx/all_notebooks_coverage.html
    :alt: Notebook Coverage

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
*QuantileLinearRegression* which trains a linear regression with *L1* norm
non-linear correlation based on decision trees, or
*QuantileMLPRegressor* a modification of scikit-learn's MLPRegressor
which trains a multi-layer perceptron with *L1* norm.
It also explores *PredictableTSNE* which trains a supervized
model to replicate *t-SNE* results or a *PiecewiseRegression*
which partitions the data before fitting a model on each bucket.

* `GitHub/mlinsights <https://github.com/sdpython/mlinsights/>`_
* `documentation <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/index.html>`_
* `Blog <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/blog/main_0000.html#ap-main-0>`_

Function ``pipeline2dot`` converts a pipeline into a graph:

::

    from mlinsights.plotting import pipeline2dot
    dot = pipeline2dot(clf, df)

.. image:: https://github.com/sdpython/mlinsights/raw/master/_doc/pipeline.png
