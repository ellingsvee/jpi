#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "mpi.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define ELEMENT_TYPE_DISPATCH(element_type, fn, ...)                           \
  switch (element_type) {                                                      \
  case ffi::F32:                                                               \
    return fn<float>(__VA_ARGS__);                                             \
  case ffi::F64:                                                               \
    return fn<double>(__VA_ARGS__);                                            \
  default:                                                                     \
    return ffi::Error::InvalidArgument("Unsupported input data type.");        \
  }

template <typename T> MPI_Datatype GetMPIDatatype() {
  if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported MPI datatype");
  }
}

#endif // UTILS_H
