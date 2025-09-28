from functools import partial
import jax
import jax.numpy as jnp
from jpi.mpi import rank, size


def _allgather_impl(x: jax.Array):
    # The input and output of the ffi_call must have the same shape and dtype
    # since we are aliasing them. For allgather, the output shape is (size * x.shape[0], ...).
    # Therefore we need to expand the input x.
    sendcount = x.shape[0]
    x_full = jnp.zeros(size * sendcount, dtype=x.dtype)
    x_full = x_full.at[0:sendcount].set(x)

    y_type = jax.ShapeDtypeStruct(x_full.shape, x_full.dtype)
    input_output_aliases = {0: 0}  # alias input and output buffers

    # NOTE: The root is unused in Allgather
    return jax.ffi.ffi_call(
        "allgather",
        (y_type,),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x_full, root=0, rank=rank, size=size, sendcount=sendcount)[0]


@partial(jax.custom_vjp)
def allgather(x: jax.Array):
    return _allgather_impl(x)


def allgather_fwd(x: jax.Array):
    return _allgather_impl(x), (x.shape[0],)


def allgather_bwd(res: tuple, g: jax.Array):
    (sendcount,) = res
    # Gradient is simply the slice of g corresponding to this rank
    start = rank * sendcount
    end = start + sendcount
    return (g[start:end],)


allgather.defvjp(allgather_fwd, allgather_bwd)
