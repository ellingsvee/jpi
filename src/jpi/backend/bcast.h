#ifndef BCAST_H
#define BCAST_H

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error BcastDispatch(int64_t root, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y);

#endif // BCAST_H