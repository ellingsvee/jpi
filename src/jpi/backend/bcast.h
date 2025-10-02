#pragma once
#ifndef BCAST_H
#define BCAST_H
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "utils.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <vector>
namespace ffi = xla::ffi;

template <typename T>
ffi::Error BcastImpl(MPI_Comm comm, int root, int numel,
                     ffi::AnyBuffer x, ffi::AnyBuffer token,
                     ffi::Result<ffi::AnyBuffer> y,
                     ffi::Result<ffi::AnyBuffer> token_out)
{
  // Get rank and size from communicator
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Get typed data pointers
  T *x_data = x.typed_data<T>(); // Not const because of MPI_IN_PLACE
  T *y_data = y->typed_data<T>();

  // Token is a dummy buffer, just alias input to output
  int32_t *token_data = token.typed_data<int32_t>();
  int32_t *token_out_data = token_out->typed_data<int32_t>();
  *token_out_data =
      *token_data; // Copy token to output (no-op in practice due to aliasing)

  // Call MPI_Reduce
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Bcast(
      static_cast<void *>(y_data),
      // y_data,
      numel,
      mpi_dtype,
      root,
      comm);
  return handle_mpi_result(ierr);
}

ffi::Error BcastDispatch(int64_t comm_handle, int64_t root, ffi::AnyBuffer x,
                         ffi::AnyBuffer token,
                         ffi::Result<ffi::AnyBuffer> y,
                         ffi::Result<ffi::AnyBuffer> token_out)
{

  // Check that input and output have same number of elements
  size_t numel = x.element_count();
  if (numel != y->element_count())
  {
    return ffi::Error::InvalidArgument(
        "Input and output must have same element count");
  }

  // Check token shapes
  if (token.element_count() != 1 || token_out->element_count() != 1)
  {
    return ffi::Error::InvalidArgument("Token must be a scalar");
  }

  // Cast to int for MPI
  int root_int = static_cast<int>(root);
  int numel_int = static_cast<int>(numel);

  // Convert handle to MPI_Comm
  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_handle));

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, BcastImpl, comm, root_int, numel_int, x, token, y, token_out);
}
#endif // BCAST_H
