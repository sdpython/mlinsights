#
# module: mlinsights.mlmodel.direct_blas_lapack
#
message(STATUS "+ CYTHON mlinsights.mlmodel.direct_blas_lapack")

cython_add_module(
  direct_blas_lapack
  ../mlinsights/mlmodel/direct_blas_lapack.pyx
  OpenMP::OpenMP_CXX)
