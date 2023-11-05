
mlinsights: tricky scikit-learn
===============================

.. image:: https://raw.github.com/sdpython/mlinsights/blob/main/_doc/_static/project_ico.png
    :target: https://github.com/sdpython/mlinsights/

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

.. image:: https://pepy.tech/badge/mlinsights
    :target: https://pypi.org/project/mlinsights/
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

*mlinsights* implements functions to get insights on machine learned
models or various kind of transforms to help manipulating
data in a single pipeline. It implements
:class:`QuantileLinearRegression <mlinsights.mlmodel.quantile_regression.QuantileLinearRegression>`
which trains a linear regression with :epkg:`L1` norm
non-linear correlation based on decision trees,
:class:`QuantileMLPRegressor
<mlinsights.mlmodel.quantile_mlpregressor.QuantileMLPRegressor>`
which is a modification of *scikit-learn's* :epkg:`MLPRegressor`
which trains a multi-layer perceptron with :epkg:`L1` norm...

.. toctree::
    :maxdepth: 1
    :caption: Documentation

    tutorial/index
    api/index
    auto_examples/index
    i_ex
    i_faq

.. toctree::
    :maxdepth: 1
    :caption: More
    
    license
    CHANGELOGS

Short example:

.. runpython::
    :showcode:
    :warningout: FutureWarning

    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from mlinsights.mlmodel import QuantileLinearRegression

    data = load_diabetes()
    X, y = data.data, data.target

    clq = QuantileLinearRegression()
    clq.fit(X, y)
    print(clq.coef_)

    clr = LinearRegression()
    clr.fit(X, y)
    print(clr.coef_)

This documentation was generated with :epkg:`scikit-learn`
version...

.. runpython::
    :showcode:

    from sklearn import __version__
    print(__version__)
