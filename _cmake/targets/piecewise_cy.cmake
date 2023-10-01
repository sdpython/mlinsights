#
# module: mlinsights.mlmodel.piecewise_tree_regression_criterion*
#
message(STATUS "+ CYTHON mlinsights.mlmodel._piecewise_tree_regression_common")

cython_add_module(
  _piecewise_tree_regression_common
  ../mlinsights/mlmodel/_piecewise_tree_regression_common.pyx
  OpenMP::OpenMP_CXX)

message(STATUS "+ CYTHON mlinsights.mlmodel.piecewise_tree_regression_criterion")

cython_add_module(
  piecewise_tree_regression_criterion
  ../mlinsights/mlmodel/piecewise_tree_regression_criterion.pyx
  OpenMP::OpenMP_CXX)

message(STATUS "+ CYTHON mlinsights.mlmodel.piecewise_tree_regression_criterion_fast")

cython_add_module(
  piecewise_tree_regression_criterion_fast
  ../mlinsights/mlmodel/piecewise_tree_regression_criterion_fast.pyx
  OpenMP::OpenMP_CXX)

message(STATUS "+ CYTHON mlinsights.mlmodel.piecewise_tree_regression_criterion_linear")

cython_add_module(
  piecewise_tree_regression_criterion_linear
  ../mlinsights/mlmodel/piecewise_tree_regression_criterion_linear.pyx
  OpenMP::OpenMP_CXX)
