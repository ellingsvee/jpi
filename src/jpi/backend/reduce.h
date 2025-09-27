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

// Helper to map integer op to MPI_Op (assumed to be in utils.h, but shown here for completeness)
// You can expand this with more operations as needed.
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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t numel = x.element_count();
  if (numel != y->element_count())
  {
    return ffi::Error::InvalidArgument(
        "Input and output must have same element count");
  }
  T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();

  ffi::Error res = handle_aliasing(x_data, y_data, rank, root);
  if (res.failure())
  {
    return res;
  }
  // Collective Reduce using the output buffer as the basis
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
  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, ReduceImpl, root, op, x, y);
}
#endif // REDUCE_H