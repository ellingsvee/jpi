#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "mpi.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

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

template <typename T>
MPI_Datatype GetMPIDatatype()
{
  if constexpr (std::is_same_v<T, float>)
  {
    return MPI_FLOAT;
  }
  else if constexpr (std::is_same_v<T, double>)
  {
    return MPI_DOUBLE;
  }
  else
  {
    static_assert(!std::is_same_v<T, T>, "Unsupported MPI datatype");
  }
}

// template <typename T>
// ffi::Error handle_aliasing(const T *in_data, T *out_data, int rank, int root, int numel)
// {
//   // Check for aliasing (e.g., donation may have triggered)
//   bool is_aliased =
//       (static_cast<const void *>(in_data) == static_cast<const void *>(out_data));
//   if (root == rank && !is_aliased)
//   {
//     // WARNING: For now we throw an error here.
//     std::memcpy(in_data, out_data, numel * sizeof(T));
//     // return ffi::Error::Internal(std::string("TRIED TO COPY"));
//   }
//   return ffi::Error::Success();
// }

ffi::Error handle_mpi_result(int ierr)
{
  if (ierr != MPI_SUCCESS)
  {
    char errstr[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(ierr, errstr, &len);
    return ffi::Error::Internal(std::string("MPI_Bcast failed: ") + errstr);
  }

  return ffi::Error::Success();
}

MPI_Op GetMPIOp(int op)
{
  switch (op)
  {
  case 0:
    return MPI_SUM;
  case 1:
    return MPI_PROD;
  case 2:
    return MPI_MIN;
  case 3:
    return MPI_MAX;
  default:
    throw std::invalid_argument("Invalid reduction op");
  }
}

#endif // UTILS_H
