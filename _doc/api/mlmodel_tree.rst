==========================
mlinsights.mlmodel (trees)
==========================

.. _blog-internal-api-impurity-improvement:

Note about potentiel issues
===========================

The main estimator `PiecewiseTreeRegressor` is based on the implementation
on new criterion. It relies on a non-public API and as such is more likely
to break. The unit test are unstable. They work when *scikit-learn*
and this package are compiled with the same set of tools. If installed
from PyPi, you can check which versions were used at compilation time.

.. runpython::
    :showcode:

    from mlinsights._config import (
        CYTHON_VERSION,
        NUMPY_VERSION,
        SCIPY_VERSION,
        SKLEARN_VERSION,
    )
    print(f"CYTHON_VERSION: {CYTHON_VERSION}")
    print(f"NUMPY_VERSION: {NUMPY_VERSION}")
    print(f"SCIPY_VERSION: {SCIPY_VERSION}")
    print(f"SKLEARN_VERSION: {SKLEARN_VERSION}")


The signature of method *impurity_improvement* has changed in version 0.24.
That's usually easy to handle two versions of *scikit-learn* even overloaded
in a class except that method is implemented in cython.
The method must be overloaded the same way with the same signature.
Tricks such as `*args` or `**kwargs` cannot be used.
The way it was handled is implemented in
PR `88 <https://github.com/sdpython/mlinsights/pull/88>`_.

Estimators
==========

PiecewiseTreeRegressor
++++++++++++++++++++++

.. autoclass:: mlinsights.mlmodel.piecewise_tree_regression.PiecewiseTreeRegressor
    :members:

Criterions
==========

The following classes require :epkg:`scikit-learn` *>= 1.3.0*,
otherwise, they do not get compiled. Section :ref:`blog-internal-api-impurity-improvement`
explains why the execution may crash.

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
