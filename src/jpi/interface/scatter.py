from functools import partial
import jax
from mpi4py import MPI
import jax.numpy as jnp


def _scatter_impl(x: jax.Array, root: int):
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    y_unsliced = jax.ffi.ffi_call(
        "scatter",
        (y_type,),
        vmap_method="sequential",
    )(x, root=root)[0]  # Unpacking the tuple at the end
    comm = MPI.COMM_WORLD
    size = comm.size
    return y_unsliced[: x.shape[0] // size]  # Slice to local part


@partial(jax.custom_vjp, nondiff_argnums=(1,))
@partial(jax.jit, static_argnums=(1,))
def scatter(x: jax.Array, root: int):
    return _scatter_impl(x, root)


def scatter_fwd(x: jax.Array, root: int):
    y = _scatter_impl(x, root)
    return y, None


def scatter_bwd(root: int, _, g: jax.Array):
    raise NotImplementedError(
        "Here we should do an allgather, but it's not implemented yet."
    )


scatter.defvjp(scatter_fwd, scatter_bwd)
