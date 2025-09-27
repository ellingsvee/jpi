#include "utils.h"
#include "bcast.h"

#include <cstdint>
#include <vector>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "mpi.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// MPI BCAST

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
ffi::Error BcastImpl(int64_t root, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t numel = x.element_count();
  if (numel != y->element_count())
  {
    return ffi::Error::InvalidArgument("Input and output must have same element count");
  }

  const T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();

  // Check for aliasing (e.g., donation may have triggered)
  bool is_aliased = (static_cast<const void *>(x_data) == static_cast<const void *>(y_data));
  if (static_cast<int>(root) == rank && !is_aliased)
  {
    // WARNING: For now we throw an error here.
    // std::memcpy(y_data, x_data, numel * sizeof(T));
    return ffi::Error::Internal(std::string("TRIED TO COPY"));
  }

  // Collective Bcast on the output buffer
  MPI_Datatype mpi_dtype = GetMPIDatatype<T>();
  int ierr = MPI_Bcast(y_data, static_cast<int>(numel), mpi_dtype, static_cast<int>(root), MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    char errstr[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(ierr, errstr, &len);
    return ffi::Error::Internal(std::string("MPI_Bcast failed: ") + errstr);
  }

  return ffi::Error::Success();
}

ffi::Error BcastDispatch(int64_t root, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y)
{
  auto dtype = x.element_type();
  ELEMENT_TYPE_DISPATCH(dtype, BcastImpl, root, x, y);
}