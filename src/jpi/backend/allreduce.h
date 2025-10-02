#pragma once
#ifndef ALLREDUCE_H
#define ALLREDUCE_H
#include "utils.h"
#include <cstdint>
#include <vector>
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

template <typename T>
ffi::Error AllReduceImpl(MPI_Comm comm, MPI_Op op, int numel, ffi::AnyBuffer x, ffi::AnyBuffer token,
                         ffi::Result<ffi::AnyBuffer> y, ffi::Result<ffi::AnyBuffer> token_out)
{
  // Get typed data pointers
  T *x_data = x.typed_data<T>(); // Not const because of MPI_IN_PLACE
  T *y_data = y->typed_data<T>();

  // Token is a dummy buffer, just alias input to output
  int32_t *token_data = token.typed_data<int32_t>();
  int32_t *token_out_data = token_out->typed_data<int32_t>();
  *token_out_data = *token_data; // Copy token to output (no-op in practice due to aliasing)

  // Call MPI_Reduce
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();

  int ierr = MPI_Allreduce(
      x_data,
      y_data,
      numel,
      mpi_dtype,
      op,
      comm);

  return handle_mpi_result(ierr);
}

ffi::Error AllReduceDispatch(int64_t comm_handle, int64_t op_handle, ffi::AnyBuffer x, ffi::AnyBuffer token,
                             ffi::Result<ffi::AnyBuffer> y, ffi::Result<ffi::AnyBuffer> token_out)
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
  int numel_int = static_cast<int>(numel);

  // Convert handle to MPI_Comm
  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_handle));
  MPI_Op op = MPI_Op_f2c(static_cast<MPI_Fint>(op_handle));

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, AllReduceImpl, comm, op, numel_int, x, token, y, token_out);
}
#endif // ALLREDUCE_H