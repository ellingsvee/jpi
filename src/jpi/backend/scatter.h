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
ffi::Error ScatterImpl(int64_t root, ffi::AnyBuffer x,
                       ffi::Result<ffi::AnyBuffer> y)
{
  // Get the rank
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get typed data pointers
  T *x_data = x.typed_data<T>(); // Not const because of MPI_IN_PLACE
  T *y_data = y->typed_data<T>();

  size_t numel = x.element_count();
  if (numel % size != 0)
  {
    return ffi::Error::InvalidArgument("Input element count must be divisible by number of processes");
  }
  int recv_count = numel / size;

  // Call MPI_Scatter
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Scatter(
      x_data,
      recv_count,
      mpi_dtype,
      y_data,
      recv_count,
      mpi_dtype,
      static_cast<int>(root),
      MPI_COMM_WORLD);

  if (ierr != MPI_SUCCESS)
  {
    char errstr[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(ierr, errstr, &len);
    return ffi::Error::Internal(std::string("MPI_Reduce failed: ") + errstr);
  }

  return ffi::Error::Success();
}

ffi::Error ScatterDispatch(int64_t root, ffi::AnyBuffer x,
                           ffi::Result<ffi::AnyBuffer> y)
{

  if (x.element_count() != y->element_count())
  {
    return ffi::Error::InvalidArgument(
        "Input and output must have same element count");
  }

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, ScatterImpl, root, x, y);
}
#endif // SCATTER_H