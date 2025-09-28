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

MPI_Op GetMPIOp(int64_t op)
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
ffi::Error ReduceImpl(int64_t root, int64_t op, ffi::AnyBuffer x,
                      ffi::Result<ffi::AnyBuffer> y)
{
  // Get the rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
  size_t numel = x.element_count();
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  MPI_Op mpi_op = GetMPIOp(op);
  void *sendbuf = (rank == static_cast<int>(root) ? MPI_IN_PLACE : static_cast<void *>(x_data));
  void *recvbuf = y_data;
  int ierr = MPI_Reduce(sendbuf, recvbuf, static_cast<int>(numel), mpi_dtype, mpi_op,
                        static_cast<int>(root), MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    char errstr[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(ierr, errstr, &len);
    return ffi::Error::Internal(std::string("MPI_Reduce failed: ") + errstr);
  }

  return ffi::Error::Success();
}

ffi::Error ReduceDispatch(int64_t root, int64_t op, ffi::AnyBuffer x,
                          ffi::Result<ffi::AnyBuffer> y)
{

  if (x.element_count() != y->element_count())
  {
    return ffi::Error::InvalidArgument(
        "Input and output must have same element count");
  }

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, ReduceImpl, root, op, x, y);
}
#endif // REDUCE_H