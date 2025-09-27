#ifndef CSR_SPMM_H
#define CSR_SPMM_H

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error CsrSpmmDispatch(int64_t n_cols, ffi::AnyBuffer Ap, ffi::AnyBuffer Aj,
                           ffi::AnyBuffer Ax, ffi::AnyBuffer Bp,
                           ffi::AnyBuffer Bj, ffi::AnyBuffer Bx,
                           ffi::Result<ffi::AnyBuffer> Cp,
                           ffi::Result<ffi::AnyBuffer> Cj,
                           ffi::Result<ffi::AnyBuffer> Cx,
                           ffi::Result<ffi::AnyBuffer> nnz);

ffi::Error CsrSpmmMaskedDispatch(int64_t n_cols, ffi::AnyBuffer Ap, ffi::AnyBuffer Aj,
                                 ffi::AnyBuffer Ax, ffi::AnyBuffer Bp,
                                 ffi::AnyBuffer Bj, ffi::AnyBuffer Bx,
                                 ffi::AnyBuffer Mp, ffi::AnyBuffer Mj,
                                 ffi::Result<ffi::AnyBuffer> Cp,
                                 ffi::Result<ffi::AnyBuffer> Cj,
                                 ffi::Result<ffi::AnyBuffer> Cx,
                                 ffi::Result<ffi::AnyBuffer> nnz);

#endif // CSR_SPMM_H