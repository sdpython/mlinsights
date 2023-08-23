#
# module: mlinsights.mltree._tree_digitize
#
message(STATUS "+ CYTHON mlinsights.mltree._tree_digitize")

cython_add_module(
  _tree_digitize
  ../mlinsights/mltree/_tree_digitize.pyx
  OpenMP::OpenMP_CXX)
