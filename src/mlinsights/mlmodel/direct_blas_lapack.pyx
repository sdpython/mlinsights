"""
@file
@brief Direct calls to libraries :epkg:`BLAS` and :epkg:`LAPACK`.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf
from libc.math cimport NAN

import numpy
cimport numpy
cimport cython
numpy.import_array()
cimport scipy.linalg.cython_lapack as cython_lapack


@cython.boundscheck(False)
@cython.wraparound(False)
def dgelss(double[:, ::1] A, double [:, ::1] B, double prec=-1.):
    """
    Finds *X* in the problem :math:`AX=B` by minimizing
    :math:`\\norm{AX - B}^2`. Uses function
    `dgels <http://www.netlib.org/lapack/explore-html/d8/dde/dgels_8f.html>`_.
    
    :param A: matrix with 2 dimensions
    :param B: matrix with 2 dimensions
    :param prec: precision
    :return: integer (INFO)
    
    INFO is:
    
    * ``= 0``: successful exit
    * ``< 0``: if INFO = -i, the i-th argument had an illegal value
    * ``> 0``: if INFO =  i, the i-th diagonal element of the
      triangular factor of A is zero, so that A does not have
      full rank; the least squares solution could not be
      computed.
      
    .. note::
        ``::1`` indicates A, B, C must be contiguous arrays.
        Arrays *A*, *B* are modified by the function.
        *B* contains the solution.

    .. exref::
        :title: Use lapack function dgelss
        
        *C* minimizes the problem :math:`\\norm{AX - B}^2`.
        
        .. runpython::
            :showcode:
        
            import numpy
            from scipy.linalg.lapack import dgelss as scipy_dgelss
            from mlinsights.mlmodel.direct_blas_lapack import dgelss
        
            A = numpy.array([[10., 1.], [12., 1.], [13., 1]])
            B = numpy.array([[20., 22., 23.]]).T
            v, x, s, rank, work, info = scipy_dgelss(A, B)
            print(x[:2])
            
            A = A.T.copy()
            info = dgelss(A, B)
            assert info == 0
            print(B[:2])
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("A and B have mismatched dimensions: %d != %d." % (A.shape[1], B.shape[0]))
    cdef int res
    cdef int rank
    with nogil:
        res = _dgelss(A, B, &rank, &prec)
    return res
    
    
cdef void copy2array2(const double* pC, double[:, ::1] C) nogil:
    """
    Copies double from a buffer to an array.
    """
    cdef size_t size = C.shape[0] * C.shape[1]
    memcpy(&C[0,0], pC, size * sizeof(double))
    
                
cdef void copy2array1(const double* pC, double[::1] C) nogil:
    """
    Copies double from a buffer to an array.
    """
    cdef size_t size = C.shape[0]
    memcpy(&C[0], pC, size * sizeof(double))
    
                
cdef int _dgelss(double[:, ::1] A, double [:, ::1] B, int* rank, const double * rcond) nogil:
    """
    Same function as :func:`dgels` but does no check.
    """
    cdef int col = A.shape[0]
    cdef int row = A.shape[1]
    cdef int info
    cdef double *pC
    cdef double *pS
    cdef int work = min(row, col) * 3 + max(max(row, col), min(row, col) * 2)
    
    pC = <double*> calloc(work, sizeof(double))
    pS = <double*> calloc(col, sizeof(double))
    
    _dgelss_noalloc(A, B, rank, rcond, pS, pC, &work, &info)

    free(pC)
    free(pS)
    return info


cdef void _dgelss_noalloc(double[:, ::1] A, double [:, ::1] B, int* rank, const double* rcond,
                          double* pS, double *pC, int* work, int* info) nogil:
    """
    Same function as :func:`dgels` but does no check.
    """
    cdef int col = A.shape[0]
    cdef int row = A.shape[1]
    cdef int nrhs = B.shape[1]
    cdef int lda = row
    cdef int ldb = row
    
    cython_lapack.dgelss(&row, &col, &nrhs,             # 1-3
                         &A[0,0], &lda, &B[0,0], &ldb,  # 4-7
                         pS, rcond, rank,               # 8-10
                         pC, work, info)                # 11-13
