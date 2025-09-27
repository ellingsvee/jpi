// #include "csr_spmm.h"
#include "bcast.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// Register the handler symbol (matches Python ffi_call target)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Bcast,
    BcastDispatch,
    ffi::Ffi::Bind()
        .Attr<int64_t>("root")
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
    return registrations; });
}
