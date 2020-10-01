import numpy
from numpy import linalg

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import util

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver



def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=False):
    """Solve the equation a x = b for x, assuming a is a triangular matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        lower (bool): Use only data contained in the lower triangle of `a`.
            Default is to use upper triangle.
        trans ({0, 1, 2, 'N', 'T', 'C'}): Type of system to solve:
            ========  =========
            trans     system
            ========  =========
            0 or 'N'  a x  = b
            1 or 'T'  a^T x = b
            2 or 'C'  a^H x = b
            ========  =========
        unit_diagonal (bool): If True, diagonal elements of `a` are assumed to
            be 1 and will not be referenced.
        overwrite_b (bool): Allow overwriting data in b (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`scipy.linalg.solve_triangular`
    """

    util._assert_cupy_array(a, b)

    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('expected square matrix')
    if len(a) != len(b):
        raise ValueError('incompatible dimensions')

    # Cast to float32 or float64
    if a.dtype.char in 'fd':
        dtype = a.dtype
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ())

    a = cupy.array(a, dtype=dtype, order='F', copy=False)
    b = cupy.array(b, dtype=dtype, order='F', copy=(not overwrite_b))

    if check_finite:
        if a.dtype.kind == 'f' and not cupy.isfinite(a).all():
            raise ValueError(
                'array must not contain infs or NaNs')
        if b.dtype.kind == 'f' and not cupy.isfinite(b).all():
            raise ValueError(
                'array must not contain infs or NaNs')

    m, n = (b.size, 1) if b.ndim == 1 else b.shape
    cublas_handle = device.get_cublas_handle()

    if dtype == 'f':
        trsm = cublas.strsm
    elif dtype == 'd':
        trsm = cublas.dtrsm
    elif dtype == 'F':
        trsm = cublas.ctrsm
    elif dtype == 'D':
        trsm = cublas.ztrsm

    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if trans == 'N':
        trans = cublas.CUBLAS_OP_N
    elif trans == 'T':
        trans = cublas.CUBLAS_OP_T
    elif trans == 'C':
        trans = cublas.CUBLAS_OP_C

    if unit_diagonal:
        diag = cublas.CUBLAS_DIAG_UNIT
    else:
        diag = cublas.CUBLAS_DIAG_NON_UNIT

    trsm(
        cublas_handle, cublas.CUBLAS_SIDE_LEFT, uplo,
        trans, diag,
        m, n, 1, a.data.ptr, m, b.data.ptr, m)
    return b

#DEPRECATED as it is supported in CUPY 8.0
#taken from https://github.com/leofang/cupy/blob/85b66e1c8cb85701f22993499e5116c2edda7bf2/cupy/linalg/decomposition.py
# def qr(a, mode='reduced'):
#     """QR decomposition.
#     Decompose a given two-dimensional matrix into ``Q * R``, where ``Q``
#     is an orthonormal and ``R`` is an upper-triangular matrix.
#     Args:
#         a (cupy.ndarray): The input matrix.
#         mode (str): The mode of decomposition. Currently 'reduced',
#             'complete', 'r', and 'raw' modes are supported. The default mode
#             is 'reduced', in which matrix ``A = (M, N)`` is decomposed into
#             ``Q``, ``R`` with dimensions ``(M, K)``, ``(K, N)``, where
#             ``K = min(M, N)``.
#     Returns:
#         cupy.ndarray, or tuple of ndarray:
#             Although the type of returned object depends on the mode,
#             it returns a tuple of ``(Q, R)`` by default.
#             For details, please see the document of :func:`numpy.linalg.qr`.
#     .. seealso:: :func:`numpy.linalg.qr`
#     """
#     if not cuda.cusolver_enabled:
#         raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

#     # TODO(Saito): Current implementation only accepts two-dimensional arrays
#     util._assert_cupy_array(a)
#     util._assert_rank2(a)

#     if mode not in ('reduced', 'complete', 'r', 'raw'):
#         if mode in ('f', 'full', 'e', 'economic'):
#             msg = 'The deprecated mode \'{}\' is not supported'.format(mode)
#             raise ValueError(msg)
#         else:
#             raise ValueError('Unrecognized mode \'{}\''.format(mode))

#     # support float32, float64, complex64, and complex128
#     if a.dtype.char in 'fdFD':
#         dtype = a.dtype.char
#     else:
#         dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

#     m, n = a.shape
#     x = a.transpose().astype(dtype, order='C', copy=True)
#     mn = min(m, n)
#     handle = device.get_cusolver_handle()
#     dev_info = cupy.empty(1, dtype=numpy.int32)
#     # compute working space of geqrf and orgqr, and solve R
#     if dtype == 'f':
#         geqrf_bufferSize = cusolver.sgeqrf_bufferSize
#         geqrf = cusolver.sgeqrf
#     elif dtype == 'd':
#         geqrf_bufferSize = cusolver.dgeqrf_bufferSize
#         geqrf = cusolver.dgeqrf
#     elif dtype == 'F':
#         geqrf_bufferSize = cusolver.cgeqrf_bufferSize
#         geqrf = cusolver.cgeqrf
#     elif dtype == 'D':
#         geqrf_bufferSize = cusolver.zgeqrf_bufferSize
#         geqrf = cusolver.zgeqrf
#     else:
#         msg = ('dtype must be float32, float64, complex64 or complex128'
#                ' (actual: {})'.format(a.dtype))
#         raise ValueError(msg)
#     buffersize = geqrf_bufferSize(handle, m, n, x.data.ptr, n)
#     workspace = cupy.empty(buffersize, dtype=dtype)
#     tau = cupy.empty(mn, dtype=dtype)
#     geqrf(handle, m, n, x.data.ptr, m,
#           tau.data.ptr, workspace.data.ptr, buffersize, dev_info.data.ptr)

#     status = int(dev_info[0])
#     if status < 0:
#         raise linalg.LinAlgError(
#             'Parameter error (maybe caused by a bug in cupy.linalg?)')

#     if mode == 'r':
#         r = x[:, :mn].transpose()
#         return util._triu(r)

#     if mode == 'raw':
#         if a.dtype.char == 'f':
#             # The original numpy.linalg.qr returns float64 in raw mode,
#             # whereas the cusolver returns float32. We agree that the
#             # following code would be inappropriate, however, in this time
#             # we explicitly convert them to float64 for compatibility.
#             return x.astype(numpy.float64), tau.astype(numpy.float64)
#         elif a.dtype.char == 'F':
#             # The same applies to complex64
#             return x.astype(numpy.complex128), tau.astype(numpy.complex128)
#         return x, tau

#     if mode == 'complete' and m > n:
#         mc = m
#         q = cupy.empty((m, m), dtype)
#     else:
#         mc = mn
#         q = cupy.empty((n, m), dtype)
#     q[:n] = x

#     # solve Q
#     if dtype == 'f':
#         orgqr_bufferSize = cusolver.sorgqr_bufferSize
#         orgqr = cusolver.sorgqr
#     elif dtype == 'd':
#         orgqr_bufferSize = cusolver.dorgqr_bufferSize
#         orgqr = cusolver.dorgqr
#     elif dtype == 'F':
#         orgqr_bufferSize = cusolver.cungqr_bufferSize
#         orgqr = cusolver.cungqr
#     elif dtype == 'D':
#         orgqr_bufferSize = cusolver.zungqr_bufferSize
#         orgqr = cusolver.zungqr
#     buffersize = orgqr_bufferSize(handle, m, mc, mn, q.data.ptr, m,
#                                   tau.data.ptr)
#     workspace = cupy.empty(buffersize, dtype=dtype)
#     orgqr(handle, m, mc, mn, q.data.ptr, m, tau.data.ptr,
#           workspace.data.ptr, buffersize, dev_info.data.ptr)

#     q = q[:mc].transpose()
#     r = x[:, :mc].transpose()
#     return q, util._triu(r)
