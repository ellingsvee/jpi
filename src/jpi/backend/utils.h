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

template <typename T>
ffi::Error handle_aliasing(const T *in_data, T *out_data, int rank, int root)
{
  // Check for aliasing (e.g., donation may have triggered)
  bool is_aliased =
      (static_cast<const void *>(in_data) == static_cast<const void *>(out_data));
  if (static_cast<int>(root) == rank && !is_aliased)
  {
    // WARNING: For now we throw an error here.
    // std::memcpy(y_data, x_data, numel * sizeof(T));
    return ffi::Error::Internal(std::string("TRIED TO COPY"));
  }
  return ffi::Error::Success();
}

#endif // UTILS_H
