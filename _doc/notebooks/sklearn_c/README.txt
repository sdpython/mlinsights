Extensions to scikit-learn involving Cython
===========================================

Experiments with :epkg:`scikit-learn` and :epkg:`cython`.
The first experiment implements a criterion for
a :epkg:`sklearn:tree:DecisionTreeRegressor`. This
code is based on the API in
`Criterion <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pxd#L56>`_
which changed in version 0.21.

