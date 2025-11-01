#pragma once
#ifndef RECV_H
#define RECV_H
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "utils.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <vector>
namespace ffi = xla::ffi;

template <typename T>
ffi::Error RecvImpl(MPI_Comm comm, int numel_per_rank, int source, int tag,
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

  // Typed pointers
  const T *x_data = x.typed_data<T>(); // const ok
  T *y_data = y->typed_data<T>();

  // Token passthrough (scalar)
  if (token.element_count() != 1 || token_out->element_count() != 1)
  {
    return ffi::Error::InvalidArgument("token and token_out must be scalars");
  }
  int32_t *token_in = token.typed_data<int32_t>();
  int32_t *token_out_ptr = token_out->typed_data<int32_t>();
  *token_out_ptr = *token_in;

  // MPI_Recv
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Recv(
      const_cast<T *>(x_data), // recvbuf
      numel_per_rank,          // recvcount (elements)
      mpi_dtype,               // recvtype
      source,                  // source
      tag,                     // tag
      comm,                    // com
      MPI_STATUS_IGNORE);
  return handle_mpi_result(ierr);
}

ffi::Error RecvDispatch(int64_t comm_handle, int64_t numel_per_rank, int64_t source, int64_t tag, ffi::AnyBuffer x,
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
  ELEMENT_TYPE_DISPATCH(dtype, RecvImpl, comm, static_cast<int>(numel_per_rank), static_cast<int>(source), static_cast<int>(tag), x, token, y, token_out);
}
#endif // RECV_H
