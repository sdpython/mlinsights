
mlinsights: tricky scikit-learn
===============================

**Links:** `github <https://github.com/sdpython/mlinsights/>`_,
`documentation <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/index.html>`_,
:ref:`README <l-README>`,
:ref:`blog <ap-main-0>`

.. image:: https://travis-ci.org/sdpython/mlinsights.svg?branch=master
    :target: https://travis-ci.org/sdpython/mlinsights
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/uj6tq445k3na7hs9?svg=true
    :target: https://ci.appveyor.com/project/sdpython/mlinsights
    :alt: Build Status Windows

.. image:: https://circleci.com/gh/sdpython/mlinsights/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/mlinsights/tree/master

.. image:: https://dev.azure.com/xavierdupre3/mlinsights/_apis/build/status/sdpython.mlinsights
    :target: https://dev.azure.com/xavierdupre3/mlinsights/

.. image:: https://badge.fury.io/py/mlinsights.svg
    :target: http://badge.fury.io/py/mlinsights

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://requires.io/github/sdpython/mlinsights/requirements.svg?branch=master
     :target: https://requires.io/github/sdpython/mlinsights/requirements/?branch=master
     :alt: Requirements Status

.. image:: https://codecov.io/github/sdpython/mlinsights/coverage.svg?branch=master
    :target: https://codecov.io/github/sdpython/mlinsights?branch=master

.. image:: http://img.shields.io/github/issues/sdpython/mlinsights.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/mlinsights/issues

.. image:: nbcov.png
    :target: http://www.xavierdupre.fr/app/mlinsights/helpsphinx/all_notebooks_coverage.html
    :alt: Notebook Coverage

*mlinsights* implements functions to get insights on machine learned
models or various kind of transforms to help manipulating
data in a single pipeline. It implements
:class:`QuantileLinearRegression <mlinsights.mlmodel.quantile_regression.QuantileLinearRegression>`
which trains a linear regression with :epkg:`L1` norm
non-linear correlation based on decision trees,
:class:`QuantileMLPRegressor <mlinsights.mlmodel.quantile_mlpregressor.QuantileMLPRegressor>`
which is a modification of *scikit-learn's* :epkg:`MLPRegressor`
which trains a multi-layer perceptron with :epkg:`L1` norm...

.. toctree::
    :maxdepth: 1

    tutorial/index
    api/index
    i_ex
    all_notebooks
    blog/blogindex
    i_index

Short example:

::

    from mlinsights.mlmodel import QuantileLinearRegression
    clq = QuantileLinearRegression()
    clq.fit(X, Y)

+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`l-modules`     |  :ref:`l-functions` | :ref:`l-classes`    | :ref:`l-methods`   | :ref:`l-staticmethods` | :ref:`l-properties`                            |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`modindex`      |  :ref:`l-EX2`       | :ref:`search`       | :ref:`l-license`   | :ref:`l-changes`       | :ref:`l-README`                                |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`genindex`      |  :ref:`l-FAQ2`      | :ref:`l-notebooks`  | :ref:`l-HISTORY`   | :ref:`l-statcode`      | `Unit Test Coverage <coverage/index.html>`_    |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
