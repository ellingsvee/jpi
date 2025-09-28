from functools import partial
import jax
from jpi.mpi import rank, size


def _bcast_impl(x: jax.Array, root: int):
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    input_output_aliases = {0: 0}  # alias input and output buffers
    return jax.ffi.ffi_call(
        "bcast",
        (y_type,),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, root=root, rank=rank, size=size)[0]  # Unpacking the tuple at the end


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def bcast(x: jax.Array, root: int):
    return _bcast_impl(x, root)


def bcast_fwd(x: jax.Array, root: int):
    y = _bcast_impl(x, root)
    return y, None  # No aux data needed for backward pass


def bcast_bwd(root: int, _, g: jax.Array):
    return (_bcast_impl(g, root),)  # Return as a tuple


bcast.defvjp(bcast_fwd, bcast_bwd)
