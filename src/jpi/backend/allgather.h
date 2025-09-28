#pragma once
#ifndef ALLGATHER_H
#define ALLGATHER_H
#include "utils.h"
#include <cstdint>
#include <vector>
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

template <typename T>
ffi::Error AllGatherImpl(int root, int rank, int size, int numel, int sendcount, ffi::AnyBuffer x,
                         ffi::Result<ffi::AnyBuffer> y)
{
  // Get typed data pointers
  T *x_data = x.typed_data<T>(); // Not const because of MPI_IN_PLACE
  T *y_data = y->typed_data<T>();

  // Call MPI_Reduce
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Allgather(
      (rank == root) ? MPI_IN_PLACE : static_cast<void *>(x_data),
      sendcount,
      mpi_dtype,
      y_data,
      sendcount,
      mpi_dtype,
      MPI_COMM_WORLD);
  return handle_mpi_result(ierr);
}

ffi::Error AllGatherDispatch(int64_t root, int64_t rank, int64_t size, int64_t sendcount, ffi::AnyBuffer x,
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
  int sendcount_int = static_cast<int>(sendcount);

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, AllGatherImpl, root_int, rank_int, size_int, numel_int, sendcount_int, x, y);
}
#endif // ALLGATHER_H