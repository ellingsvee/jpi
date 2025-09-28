from functools import partial
import jax
from jpi.interface.bcast import _bcast_impl
from jpi.mpi import rank, size, root


def _allreduce_impl(x: jax.Array, op: int):
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)

    # NOTE: The root is unused in Allreduce
    return jax.ffi.ffi_call(
        "allreduce",
        (y_type,),
        vmap_method="sequential",
    )(x, root=root, rank=rank, size=size, op=op)[0]


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def allreduce(x: jax.Array, op: int):
    return _allreduce_impl(x, op)


def allreduce_fwd(x: jax.Array, op: int):
    y = _allreduce_impl(x, op)
    return y, (x, op)


def allreduce_bwd(op: int, res: tuple, g: jax.Array):
    x, op = res
    if op == 0:  # MPI_SUM
        # For sum, gradient is broadcast from root to all ranks
        return (_bcast_impl(g, 0),)
    # elif op == 1:  # MPI_PROD
    #     # For product, gradient is y / x_i (on each rank)
    #     y = _reduce_impl(x, root, op)
    #     # Broadcast y to all ranks to compute gradient
    #     y = _bcast_impl(y, root)
    #     grad = jnp.where(x != 0, y / x, 0)  # Avoid division by zero
    #     return (grad,)
    # elif op in (2, 3):  # MPI_MIN, MPI_MAX
    #     # For min/max, gradient is 1 where x is min/max, 0 elsewhere
    #     y = _reduce_impl(x, root, op)
    #     # Broadcast y to all ranks
    #     y = _bcast_impl(y, root)
    #     grad = jnp.where(x == y, 1.0, 0.0)
    #     return (grad,)
    else:
        raise ValueError(f"Unsupported op: {op}")


allreduce.defvjp(allreduce_fwd, allreduce_bwd)
