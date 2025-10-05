#pragma once
#ifndef GATHER_H
#define GATHER_H
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "utils.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <vector>
namespace ffi = xla::ffi;

template <typename T>
ffi::Error GatherImpl(MPI_Comm comm, int root, int numel_per_rank,
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

  // Verify output length: must equal numel_per_rank * size
  long expected = static_cast<long>(numel_per_rank) * static_cast<long>(size);
  if (static_cast<long>(y->element_count()) != expected)
  {
    return ffi::Error::InvalidArgument("y->element_count() must be size * numel_per_rank");
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

  // MPI_Gather: each rank contributes numel_per_rank elements
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Gather(
      const_cast<T *>(x_data), // sendbuf
      numel_per_rank,          // sendcount (elements)
      mpi_dtype,               // sendtype
      y_data,                  // recvbuf (all ranks)
      numel_per_rank,          // recvcount (per-rank)
      mpi_dtype,               // recvtype
      root,
      comm);
  return handle_mpi_result(ierr);
}

ffi::Error GatherDispatch(int64_t comm_handle, int64_t root, int64_t numel_per_rank, ffi::AnyBuffer x,
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
  ELEMENT_TYPE_DISPATCH(dtype, GatherImpl, comm, static_cast<int>(root), static_cast<int>(numel_per_rank), x, token, y, token_out);
}
#endif // GATHER_H
