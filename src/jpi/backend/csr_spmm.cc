#include "csr_spmm.h"

#include <cstdint>
#include <vector>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

#define ELEMENT_TYPE_DISPATCH(element_type, fn, ...)                    \
  switch (element_type)                                                 \
  {                                                                     \
  case ffi::F32:                                                        \
    return fn<float>(__VA_ARGS__);                                      \
  case ffi::F64:                                                        \
    return fn<double>(__VA_ARGS__);                                     \
  default:                                                              \
    return ffi::Error::InvalidArgument("Unsupported input data type."); \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The "standard" CSR SpMM algorithm

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T1, class T2>
bool kv_pair_less(const std::pair<T1, T2> &x, const std::pair<T1, T2> &y)
{
  return x.first < y.first;
}

template <typename T>
void CsrSortIndices(int64_t n_rows, int64_t *cp_data, int64_t *cj_data,
                    T *cx_data)
{
  std::vector<std::pair<int64_t, T>> temp;

  for (int64_t i = 0; i < n_rows; i++)
  {
    int64_t row_start = cp_data[i];
    int64_t row_end = cp_data[i + 1];

    temp.resize(row_end - row_start);
    for (int64_t jj = row_start, n = 0; jj < row_end; jj++, n++)
    {
      temp[n].first = cj_data[jj];
      temp[n].second = cx_data[jj];
    }

    std::sort(temp.begin(), temp.end(), kv_pair_less<int64_t, T>);

    for (int64_t jj = row_start, n = 0; jj < row_end; jj++, n++)
    {
      cj_data[jj] = temp[n].first;
      cx_data[jj] = temp[n].second;
    }
  }
}

template <typename T>
ffi::Error
CsrSpmmImpl(int64_t n_rows, int64_t n_cols, int64_t max_nnz, ffi::AnyBuffer Ap,
            ffi::AnyBuffer Aj, ffi::AnyBuffer Ax, ffi::AnyBuffer Bp,
            ffi::AnyBuffer Bj, ffi::AnyBuffer Bx,
            ffi::Result<ffi::AnyBuffer> Cp, ffi::Result<ffi::AnyBuffer> Cj,
            ffi::Result<ffi::AnyBuffer> Cx, ffi::Result<ffi::AnyBuffer> nnz)
{
  const int64_t *ap_data = Ap.typed_data<int64_t>();
  const int64_t *aj_data = Aj.typed_data<int64_t>();
  const T *ax_data = Ax.typed_data<T>();
  const int64_t *bp_data = Bp.typed_data<int64_t>();
  const int64_t *bj_data = Bj.typed_data<int64_t>();
  const T *bx_data = Bx.typed_data<T>();
  int64_t *cp_data = Cp->typed_data<int64_t>();
  int64_t *cj_data = Cj->typed_data<int64_t>();
  T *cx_data = Cx->typed_data<T>();

  std::vector<int64_t> next(n_cols, -1);
  std::vector<T> sums(n_cols, T(0));
  int64_t nnz_loc = 0;
  cp_data[0] = 0;
  for (int64_t i = 0; i < n_rows; i++)
  {
    int64_t head = -2;
    int64_t length = 0;
    int64_t jj_start = ap_data[i];
    int64_t jj_end = ap_data[i + 1];
    for (int64_t jj = jj_start; jj < jj_end; jj++)
    {
      int64_t j = aj_data[jj];
      T v = ax_data[jj];
      int64_t kk_start = bp_data[j];
      int64_t kk_end = bp_data[j + 1];
      for (int64_t kk = kk_start; kk < kk_end; kk++)
      {
        int64_t k = bj_data[kk];
        sums[k] += v * bx_data[kk];
        if (next[k] == -1)
        {
          next[k] = head;
          head = k;
          length++;
        }
      }
    }
    for (int64_t jj = 0; jj < length; jj++)
    {
      if (sums[head] != T(0))
      {
        if (nnz_loc >= max_nnz)
        {
          return ffi::Error::InvalidArgument(
              "Output buffer too small for CSR nnz");
        }
        cj_data[nnz_loc] = head;
        cx_data[nnz_loc] = sums[head];
        nnz_loc++;
      }
      int64_t temp = head;
      head = next[head];
      next[temp] = -1; // clear arrays
      sums[temp] = T(0);
    }
    cp_data[i + 1] = nnz_loc;
  }

  // CsrSortIndices(n_rows, cp_data, cj_data,
  //                cx_data); // Ensure indices are sorted

  nnz->typed_data<int64_t>()[0] = nnz_loc; // Update the nnz output
  return ffi::Error::Success();
}

ffi::Error CsrSpmmDispatch(int64_t n_cols, ffi::AnyBuffer Ap, ffi::AnyBuffer Aj,
                           ffi::AnyBuffer Ax, ffi::AnyBuffer Bp,
                           ffi::AnyBuffer Bj, ffi::AnyBuffer Bx,
                           ffi::Result<ffi::AnyBuffer> Cp,
                           ffi::Result<ffi::AnyBuffer> Cj,
                           ffi::Result<ffi::AnyBuffer> Cx,
                           ffi::Result<ffi::AnyBuffer> nnz)
{
  if (Ap.element_type() != ffi::S64 || Aj.element_type() != ffi::S64 ||
      Bp.element_type() != ffi::S64 || Bj.element_type() != ffi::S64 ||
      Cp->element_type() != ffi::S64 || Cj->element_type() != ffi::S64)
  {
    return ffi::Error::InvalidArgument("Index buffers must be S64");
  }
  if (Ax.element_type() != Bx.element_type() ||
      Ax.element_type() != Cx->element_type())
  {
    return ffi::Error::InvalidArgument("Data buffers must have matching types");
  }
  auto dtype = Ax.element_type();
  if (Ap.dimensions().size() != 1 || Aj.dimensions().size() != 1 ||
      Ax.dimensions().size() != 1 || Bp.dimensions().size() != 1 ||
      Bj.dimensions().size() != 1 || Bx.dimensions().size() != 1 ||
      Cp->dimensions().size() != 1 || Cj->dimensions().size() != 1 ||
      Cx->dimensions().size() != 1)
  {
    return ffi::Error::InvalidArgument("All buffers must be 1D arrays");
  }
  int64_t n_rows = Ap.dimensions()[0] - 1;
  if (n_rows < 0)
  {
    return ffi::Error::InvalidArgument("Invalid n_rows from Ap size");
  }
  if (Cp->dimensions()[0] != Ap.dimensions()[0])
  {
    return ffi::Error::InvalidArgument("Cp must match Ap size");
  }
  int64_t max_nnz = Cj->dimensions()[0];
  if (Cx->dimensions()[0] != max_nnz)
  {
    return ffi::Error::InvalidArgument("Cj and Cx must have matching sizes");
  }
  ELEMENT_TYPE_DISPATCH(dtype, CsrSpmmImpl, n_rows, n_cols, max_nnz, Ap, Aj, Ax,
                        Bp, Bj, Bx, Cp, Cj, Cx, nnz);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The "masked" CSR SpMM algorithm

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
ffi::Error CsrSpmmMaskedImpl(
    int64_t n_rows, int64_t n_cols, int64_t max_nnz, ffi::AnyBuffer Ap,
    ffi::AnyBuffer Aj, ffi::AnyBuffer Ax, ffi::AnyBuffer Bp, ffi::AnyBuffer Bj,
    ffi::AnyBuffer Bx, ffi::AnyBuffer Mp, ffi::AnyBuffer Mj, // Mask matrix
    ffi::Result<ffi::AnyBuffer> Cp, ffi::Result<ffi::AnyBuffer> Cj,
    ffi::Result<ffi::AnyBuffer> Cx, ffi::Result<ffi::AnyBuffer> nnz)
{
  const int64_t *ap_data = Ap.typed_data<int64_t>();
  const int64_t *aj_data = Aj.typed_data<int64_t>();
  const T *ax_data = Ax.typed_data<T>();
  const int64_t *bp_data = Bp.typed_data<int64_t>();
  const int64_t *bj_data = Bj.typed_data<int64_t>();
  const T *bx_data = Bx.typed_data<T>();
  const int64_t *mp_data = Mp.typed_data<int64_t>();
  const int64_t *mj_data = Mj.typed_data<int64_t>();

  int64_t *cp_data = Cp->typed_data<int64_t>();
  int64_t *cj_data = Cj->typed_data<int64_t>();
  T *cx_data = Cx->typed_data<T>();

  // Reuse the original's efficient data structures
  std::vector<bool> is_mask_col(
      n_cols, false); // Mark which columns are in current row's mask
  std::vector<T> sums(n_cols, T(0));

  int64_t nnz_loc = 0;
  cp_data[0] = 0;

  for (int64_t i = 0; i < n_rows; i++)
  {
    // Mark mask columns for this row
    int64_t mask_start = mp_data[i];
    int64_t mask_end = mp_data[i + 1];

    for (int64_t mm = mask_start; mm < mask_end; mm++)
    {
      is_mask_col[mj_data[mm]] = true;
    }

    // Perform sparse matrix multiplication (same as original)
    int64_t jj_start = ap_data[i];
    int64_t jj_end = ap_data[i + 1];

    for (int64_t jj = jj_start; jj < jj_end; jj++)
    {
      int64_t j = aj_data[jj];
      T v = ax_data[jj];

      int64_t kk_start = bp_data[j];
      int64_t kk_end = bp_data[j + 1];

      for (int64_t kk = kk_start; kk < kk_end; kk++)
      {
        int64_t k = bj_data[kk];
        if (is_mask_col[k])
        { // Only accumulate if in mask
          sums[k] += v * bx_data[kk];
        }
      }
    }

    // Output all mask columns (in mask order, including explicit zeros)
    for (int64_t mm = mask_start; mm < mask_end; mm++)
    {
      int64_t mask_col = mj_data[mm];

      if (nnz_loc >= max_nnz)
      {
        return ffi::Error::InvalidArgument(
            "Output buffer too small for CSR nnz");
      }

      cj_data[nnz_loc] = mask_col;
      cx_data[nnz_loc] = sums[mask_col]; // Includes explicit zeros
      nnz_loc++;
    }

    // Clean up for next row (only clear mask columns)
    for (int64_t mm = mask_start; mm < mask_end; mm++)
    {
      int64_t mask_col = mj_data[mm];
      is_mask_col[mask_col] = false;
      sums[mask_col] = T(0);
    }

    cp_data[i + 1] = nnz_loc;
  }

  nnz->typed_data<int64_t>()[0] = nnz_loc;
  return ffi::Error::Success();
}

ffi::Error CsrSpmmMaskedDispatch(
    int64_t n_cols, ffi::AnyBuffer Ap, ffi::AnyBuffer Aj, ffi::AnyBuffer Ax,
    ffi::AnyBuffer Bp, ffi::AnyBuffer Bj, ffi::AnyBuffer Bx, ffi::AnyBuffer Mp,
    ffi::AnyBuffer Mj, ffi::Result<ffi::AnyBuffer> Cp,
    ffi::Result<ffi::AnyBuffer> Cj, ffi::Result<ffi::AnyBuffer> Cx,
    ffi::Result<ffi::AnyBuffer> nnz)
{
  if (Ap.element_type() != ffi::S64 || Aj.element_type() != ffi::S64 ||
      Bp.element_type() != ffi::S64 || Bj.element_type() != ffi::S64 ||
      Cp->element_type() != ffi::S64 || Cj->element_type() != ffi::S64)
  {
    return ffi::Error::InvalidArgument("Index buffers must be S64");
  }
  if (Ax.element_type() != Bx.element_type() ||
      Ax.element_type() != Cx->element_type())
  {
    return ffi::Error::InvalidArgument("Data buffers must have matching types");
  }
  auto dtype = Ax.element_type();
  if (Ap.dimensions().size() != 1 || Aj.dimensions().size() != 1 ||
      Ax.dimensions().size() != 1 || Bp.dimensions().size() != 1 ||
      Bj.dimensions().size() != 1 || Bx.dimensions().size() != 1 ||
      Cp->dimensions().size() != 1 || Cj->dimensions().size() != 1 ||
      Cx->dimensions().size() != 1)
  {
    return ffi::Error::InvalidArgument("All buffers must be 1D arrays");
  }
  int64_t n_rows = Ap.dimensions()[0] - 1;
  if (n_rows < 0)
  {
    return ffi::Error::InvalidArgument("Invalid n_rows from Ap size");
  }
  if (Cp->dimensions()[0] != Ap.dimensions()[0])
  {
    return ffi::Error::InvalidArgument("Cp must match Ap size");
  }
  int64_t max_nnz = Cj->dimensions()[0];
  if (Cx->dimensions()[0] != max_nnz)
  {
    return ffi::Error::InvalidArgument("Cj and Cx must have matching sizes");
  }

  // Make sure the mask is valid
  if (Mp.element_type() != ffi::S64 || Mj.element_type() != ffi::S64)
  {
    return ffi::Error::InvalidArgument("Mask buffers must be S64");
  }
  if (Mp.dimensions().size() != 1 || Mj.dimensions().size() != 1)
  {
    return ffi::Error::InvalidArgument("Mask buffers must be 1D arrays");
  }
  // if (Mp.dimensions()[0] != n_rows + 1 || Mj.dimensions()[0] != n_cols + 1)
  // {
  //   return ffi::Error::InvalidArgument("Mask buffers must match the size of
  //   Ap and Bp");
  // }

  ELEMENT_TYPE_DISPATCH(dtype, CsrSpmmMaskedImpl, n_rows, n_cols, max_nnz, Ap,
                        Aj, Ax, Bp, Bj, Bx, Mp, Mj, Cp, Cj, Cx, nnz);
}
