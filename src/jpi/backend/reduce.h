#pragma once
#ifndef REDUCE_H
#define REDUCE_H
#include "utils.h"
#include <cstdint>
#include <vector>
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

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

template <typename T>
ffi::Error ReduceImpl(int root, int rank, int size, int numel, int op, ffi::AnyBuffer x,
                      ffi::Result<ffi::AnyBuffer> y)
{
  // Get typed data pointers
  T *x_data = x.typed_data<T>(); // Not const because of MPI_IN_PLACE
  T *y_data = y->typed_data<T>();

  // Check for aliasing
  ffi::Error res = handle_aliasing(x_data, y_data, rank, root);
  if (res.failure())
  {
    return res;
  }

  // Call MPI_Reduce
  MPI_Op mpi_op = GetMPIOp(op);
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Reduce(
      (rank == root) ? MPI_IN_PLACE : x_data,
      y_data,
      numel,
      mpi_dtype,
      mpi_op,
      root,
      MPI_COMM_WORLD);
  return handle_mpi_result(ierr);
}

ffi::Error ReduceDispatch(int64_t root, int64_t rank, int64_t size, int64_t op, ffi::AnyBuffer x,
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
  int op_int = static_cast<int>(op);

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, ReduceImpl, root_int, rank_int, size_int, numel_int, op_int, x, y);
}
#endif // REDUCE_H