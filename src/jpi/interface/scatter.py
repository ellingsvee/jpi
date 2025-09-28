from functools import partial
import jax
from jpi.mpi import rank, size


def _scatter_impl(x: jax.Array, root: int):
    if x.shape[0] % size != 0:
        raise ValueError(
            f"x.shape[0] ({x.shape[0]}) must be divisible by number of processes ({size})"
        )
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    y_unsliced = jax.ffi.ffi_call(
        "scatter",
        (y_type,),
        vmap_method="sequential",
    )(x, root=root, rank=rank, size=size)[0]
    return y_unsliced[: x.shape[0] // size]  # Slice to local part


@partial(jax.custom_vjp, nondiff_argnums=(1,))
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
