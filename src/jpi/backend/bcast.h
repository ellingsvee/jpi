#pragma once
#ifndef BCAST_H
#define BCAST_H

#include "utils.h"
#include <cstdint>
#include <vector>

#include "mpi.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <typename T>
ffi::Error BcastImpl(int root, int rank, int size, int numel, ffi::AnyBuffer x,
                     ffi::Result<ffi::AnyBuffer> y)
{
  // Get typed data pointers
  const T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();

  // Check for aliasing (e.g., donation may have triggered)
  ffi::Error res = handle_aliasing(x_data, y_data, rank, root);
  if (res.failure())
  {
    return res;
  }

  // Call MPI_Bcast
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Bcast(
      y_data,
      numel,
      mpi_dtype,
      root,
      MPI_COMM_WORLD);

  return handle_mpi_result(ierr);
}

ffi::Error BcastDispatch(int64_t root, int64_t rank, int64_t size, ffi::AnyBuffer x,
                         ffi::Result<ffi::AnyBuffer> y)
{

  // Check that input and output have same number of elements
  size_t numel = x.element_count();
  if (numel != y->element_count())
  {
    return ffi::Error::InvalidArgument(
        "Input and output must have same element count");
  }

  // Cast to int for MPI
  int root_int = static_cast<int>(root);
  int rank_int = static_cast<int>(rank);
  int size_int = static_cast<int>(size);
  int numel_int = static_cast<int>(numel);

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, BcastImpl, root_int, rank_int, size_int, numel_int, x, y);
}

#endif // BCAST_H
