#include "csr_spmm.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

XLA_FFI_DEFINE_HANDLER_SYMBOL(CsrSpmm, CsrSpmmDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("n_cols")
                                  .Arg<ffi::AnyBuffer>() // Ap
                                  .Arg<ffi::AnyBuffer>() // Aj
                                  .Arg<ffi::AnyBuffer>() // Ax
                                  .Arg<ffi::AnyBuffer>() // Bp
                                  .Arg<ffi::AnyBuffer>() // Bj
                                  .Arg<ffi::AnyBuffer>() // Bx
                                  .Ret<ffi::AnyBuffer>() // Cp
                                  .Ret<ffi::AnyBuffer>() // Cj
                                  .Ret<ffi::AnyBuffer>() // Cx
                                  .Ret<ffi::AnyBuffer>() // nnz
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn)
{
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_core, m)
{
    m.def("registrations", []()
          {
    nb::dict registrations;
    registrations["csr_spmm"] = EncapsulateFfiHandler(CsrSpmm);
    return registrations; });
}
