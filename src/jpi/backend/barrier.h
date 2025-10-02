#pragma once
#ifndef BARRIER_H
#define BARRIER_H
#include "mpi.h"
#include "nanobind/nanobind.h"
#include "utils.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <vector>
namespace ffi = xla::ffi;

ffi::Error BarrierImpl(MPI_Comm comm, ffi::AnyBuffer token, ffi::Result<ffi::AnyBuffer> token_out)
{
  // Token is a dummy buffer, just alias input to output
  int32_t *token_data = token.typed_data<int32_t>();
  int32_t *token_out_data = token_out->typed_data<int32_t>();
  *token_out_data =
      *token_data; // Copy token to output (no-op in practice due to aliasing)

  // Call MPI_Barrier
  int ierr = MPI_Barrier(comm);
  return handle_mpi_result(ierr);
}

ffi::Error BarrierDispatch(int64_t comm_handle, ffi::AnyBuffer token, ffi::Result<ffi::AnyBuffer> token_out)
{
  // Convert handle to MPI_Comm
  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_handle));

  // auto dtype = ffi::F32; // Hardcoded for now
  // ELEMENT_TYPE_DISPATCH(dtype, BarrierImpl, comm, token, token_out);
  return BarrierImpl(comm, token, token_out);
}
#endif // BARRIER_H
