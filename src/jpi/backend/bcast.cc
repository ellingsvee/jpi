#include "bcast.h"

#include <cstdint>
#include <vector>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// The MPI
#include "mpi.h"

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

// MPI BCAST

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MPI_ELEMENT_TYPE_DISPATCH(element_type, fn, ...)                \
  switch (element_type)                                                 \
  {                                                                     \
  case ffi::F32:                                                        \
    return fn<float>(__VA_ARGS__);                                      \
  case ffi::F64:                                                        \
    return fn<double>(__VA_ARGS__);                                     \
  default:                                                              \
    return ffi::Error::InvalidArgument("Unsupported input data type."); \
  }

template <typename T>
ffi::Error
BcastImpl(
    int64_t root, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y)
{
  // MPI_Comm comm = *static_cast<MPI_Comm *>(comm_ptr);

  const T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t numel = x.element_count();

  if (static_cast<int>(root) == rank)
  {
    // On root: Copy input to output before broadcast
    std::memcpy(y_data, x_data, numel * sizeof(float)); // WARNING: NOT IDEAL
  } // On non-root: y_data is uninitialized (overwritten by Bcast)

  // Perform collective broadcast on y_data (root sends, all receive/modify)
  int ierr = MPI_Bcast(y_data, static_cast<int>(numel), MPI_FLOAT, static_cast<int>(root), MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    char errstr[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(ierr, errstr, &len);
    return ffi::Error::Internal(std::string("MPI_Bcast failed: ") + errstr);
  }

  return ffi::Error::Success();
}

ffi::Error BcastDispatch(int64_t root, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y)
{
  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, BcastImpl, root, x, y);
}