import time

from functools import partial
import jax
import jax.numpy as jnp

# Assume sparsejax is a module with FFI registrations
# from sparsejax._csr._csr_to_other import csr_to_csc
# from sparsejax._csr._csr_utils import csr_sorting


def _bcast_impl(x, root):
    y_shape = x.shape
    y_type = jax.ShapeDtypeStruct(y_shape, jnp.float32)
    return jax.ffi.ffi_call(
        "bcast",
        (y_type,),
        vmap_method="sequential",
    )(x, root=root)


# @partial(jax.custom_vjp)
# @partial(jax.jit, static_argnums=(6, 7, 8))

@partial(jax.jit, static_argnums=(1,))
def bcast(x, root):
    return _bcast_impl(x, root)
