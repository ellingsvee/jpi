#pragma once
#ifndef SCATTER_H
#define SCATTER_H
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "utils.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <vector>
namespace ffi = xla::ffi;

template <typename T>
ffi::Error ScatterImpl(MPI_Comm comm, int root, int numel_per_rank,
                       ffi::AnyBuffer x, ffi::AnyBuffer token,
                       ffi::Result<ffi::AnyBuffer> y,
                       ffi::Result<ffi::AnyBuffer> token_out)
{
  // Get rank and size from communicator
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // // Get typed data pointers
  // T *x_data = x.typed_data<T>();
  // T *y_data = y->typed_data<T>();

  // // Token is a dummy buffer, just alias input to output
  // int32_t *token_data = token.typed_data<int32_t>();
  // int32_t *token_out_data = token_out->typed_data<int32_t>();
  // *token_out_data =
  //     *token_data; // Copy token to output (no-op in practice due to aliasing)

  // // Compute recv count. Fail if not divisible.
  // int recv_count = numel / size;
  // if (numel % size != 0)
  // {
  //   return ffi::Error::InvalidArgument("Input element count must be divisible by number of processes");
  // }

  // // Call MPI_Scatter
  // MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  // int ierr = MPI_Scatter(
  //     x_data,
  //     recv_count,
  //     mpi_dtype,
  //     y_data,
  //     recv_count,
  //     mpi_dtype,
  //     root,
  //     MPI_COMM_WORLD);
  // return handle_mpi_result(ierr);

  // Token passthrough
  int32_t *token_in = token.typed_data<int32_t>();
  int32_t *token_out_ptr = token_out->typed_data<int32_t>();
  *token_out_ptr = *token_in;

  // Each rank receives numel/size elements
  if (numel_per_rank % size != 0)
  {
    return ffi::Error::InvalidArgument("numel must be divisible by comm size");
  }
  int recv_count = numel_per_rank / size;

  // Input buffer: only root actually provides data
  T *x_data = nullptr;
  if (rank == root)
  {
    x_data = x.typed_data<T>();
  }

  // Output buffer (each rank gets recv_count elems)
  T *y_data = y->typed_data<T>();

  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Scatter(
      x_data, recv_count, mpi_dtype,
      y_data, recv_count, mpi_dtype,
      root, comm);

  return handle_mpi_result(ierr);
}

ffi::Error ScatterDispatch(int64_t comm_handle, int64_t root, int64_t numel_per_rank,
                           ffi::AnyBuffer x,
                           ffi::AnyBuffer token,
                           ffi::Result<ffi::AnyBuffer> y,
                           ffi::Result<ffi::AnyBuffer> token_out)
{

  // Token scalar check
  if (token.element_count() != 1 || token_out->element_count() != 1)
  {
    return ffi::Error::InvalidArgument("Token must be a scalar");
  }

  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_handle));

  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, ScatterImpl, comm,
                        static_cast<int>(root),
                        static_cast<int>(numel_per_rank),
                        x, token, y, token_out);
}
#endif // SCATTER_H
