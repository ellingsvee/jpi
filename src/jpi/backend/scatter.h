#pragma once
#ifndef SCATTER_H
#define SCATTER_H
#include "utils.h"
#include <cstdint>
#include <vector>
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

template <typename T>
ffi::Error ScatterImpl(int root, int rank, int size, int numel, ffi::AnyBuffer x,
                       ffi::Result<ffi::AnyBuffer> y)
{
  // Get typed data pointers
  const T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();

  // Compute recv count. Fail if not divisible.
  int recv_count = numel / size;
  if (numel % size != 0)
  {
    return ffi::Error::InvalidArgument("Input element count must be divisible by number of processes");
  }

  // Call MPI_Scatter
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Scatter(
      x_data,
      recv_count,
      mpi_dtype,
      y_data,
      recv_count,
      mpi_dtype,
      root,
      MPI_COMM_WORLD);

  return handle_mpi_result(ierr);
}

ffi::Error ScatterDispatch(int64_t root, int64_t rank, int64_t size, ffi::AnyBuffer x,
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
  ELEMENT_TYPE_DISPATCH(dtype, ScatterImpl, root_int, rank_int, size_int, numel_int, x, y);
}
#endif // SCATTER_H