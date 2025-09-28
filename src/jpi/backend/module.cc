#include "bcast.h"
#include "reduce.h"
#include "scatter.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

XLA_FFI_DEFINE_HANDLER_SYMBOL(Bcast, BcastDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("root")
                                  .Attr<int64_t>("rank")
                                  .Attr<int64_t>("size")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
);
XLA_FFI_DEFINE_HANDLER_SYMBOL(Reduce, ReduceDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("root")
                                  .Attr<int64_t>("rank")
                                  .Attr<int64_t>("size")
                                  .Attr<int64_t>("op")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(Scatter, ScatterDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("root")
                                  .Attr<int64_t>("rank")
                                  .Attr<int64_t>("size")
                                  .Arg<ffi::AnyBuffer>() // Input buffer x
                                  .Ret<ffi::AnyBuffer>() // Output buffer y
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
    registrations["reduce"] = EncapsulateFfiHandler(Reduce);
    registrations["scatter"] = EncapsulateFfiHandler(Scatter);
    return registrations; });
}
