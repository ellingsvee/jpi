#include "allgather.h"
#include "allreduce.h"
#include "bcast.h"
#include "barrier.h"
#include "gather.h"
#include "scatter.h"

#include "nanobind/nanobind.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

XLA_FFI_DEFINE_HANDLER_SYMBOL(Bcast, BcastDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("comm_handle")
                                  .Attr<int64_t>("root")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Arg<ffi::AnyBuffer>() // Input token
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
                                  .Ret<ffi::AnyBuffer>() // Output token
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllGather, AllGatherDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("comm_handle")
                                  .Attr<int64_t>("numel_per_rank")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Arg<ffi::AnyBuffer>() // Input token
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
                                  .Ret<ffi::AnyBuffer>() // Output token
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(AllReduce, AllReduceDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("comm_handle")
                                  .Attr<int64_t>("op_handle")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Arg<ffi::AnyBuffer>() // Input token
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
                                  .Ret<ffi::AnyBuffer>() // Output token
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(Barrier, BarrierDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("comm_handle")
                                  .Arg<ffi::AnyBuffer>() // Input token
                                  .Ret<ffi::AnyBuffer>() // Output token
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(Gather, GatherDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("comm_handle")
                                  .Attr<int64_t>("root")
                                  .Attr<int64_t>("sendcount")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Arg<ffi::AnyBuffer>() // Input token
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
                                  .Ret<ffi::AnyBuffer>() // Output token
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(Scatter, ScatterDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("comm_handle")
                                  .Attr<int64_t>("root")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Arg<ffi::AnyBuffer>() // Input token
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
                                  .Ret<ffi::AnyBuffer>() // Output token
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn)
{
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(backend, m)
{
    m.def("registrations", []()
          {
    nb::dict registrations;
    registrations["bcast"] = EncapsulateFfiHandler(Bcast);
    registrations["allgather"] = EncapsulateFfiHandler(AllGather);
    registrations["allreduce"] = EncapsulateFfiHandler(AllReduce);
    registrations["barrier"] = EncapsulateFfiHandler(Barrier);
    registrations["gather"] = EncapsulateFfiHandler(Gather);
    registrations["scatter"] = EncapsulateFfiHandler(Scatter);
    return registrations; });
}
