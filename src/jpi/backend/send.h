#pragma once
#ifndef SEND_H
#define SEND_H
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "utils.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <vector>
namespace ffi = xla::ffi;

template <typename T>
ffi::Error SendImpl(MPI_Comm comm, int numel_per_rank, int dest, int tag,
                      ffi::AnyBuffer x, ffi::AnyBuffer token,
                      ffi::Result<ffi::AnyBuffer> y,
                      ffi::Result<ffi::AnyBuffer> token_out)
{
  // Get rank and size from communicator
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // sanity: input element count must equal numel_per_rank
  if (static_cast<int>(x.element_count()) != numel_per_rank)
  {
    return ffi::Error::InvalidArgument("x.element_count() != numel_per_rank");
  }

  // sanity: output element count must equal numel_per_rank
  if (static_cast<int>(y->element_count()) != numel_per_rank)
  {
    return ffi::Error::InvalidArgument("y->element_count() != numel_per_rank");
  }

  // When using input_output_aliases, y points to the same buffer as x
  // Verify output length matches input
  if (y->element_count() != x.element_count())
  {
    return ffi::Error::InvalidArgument("y->element_count() must equal x.element_count()");
  }

  // Typed pointers - y_data and x_data point to the same memory when aliased
  const T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();

  // Token passthrough (scalar)
  if (token.element_count() != 1 || token_out->element_count() != 1)
  {
    return ffi::Error::InvalidArgument("token and token_out must be scalars");
  }
  int32_t *token_in = token.typed_data<int32_t>();
  int32_t *token_out_ptr = token_out->typed_data<int32_t>();
  *token_out_ptr = *token_in;

  // MPI_Send
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Send(
      const_cast<T *>(x_data), // sendbuf
      numel_per_rank,          // sendcount (elements)
      mpi_dtype,               // sendtype
      dest,                    // dest
      tag,                     // tag
      comm);
  return handle_mpi_result(ierr);
}

ffi::Error SendDispatch(int64_t comm_handle, int64_t numel_per_rank, int64_t dest, int64_t tag, ffi::AnyBuffer x,
                          ffi::AnyBuffer token,
                          ffi::Result<ffi::AnyBuffer> y,
                          ffi::Result<ffi::AnyBuffer> token_out)
{

  if (token.element_count() != 1 || token_out->element_count() != 1)
  {
    return ffi::Error::InvalidArgument("Token must be scalar");
  }

  // Convert handle to MPI_Comm
  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_handle));

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, SendImpl, comm, static_cast<int>(numel_per_rank), static_cast<int>(dest), static_cast<int>(tag), x, token, y, token_out);
}
#endif // SEND_H
