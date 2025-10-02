from functools import partial
import jax
from jpi.comm import get_default_comm


def _allreduce_impl(x: jax.Array, token: jax.Array, comm, op):
    # op = unpack_hashable(op)

    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)

    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {1: 1}  # alias input and output buffers

    result, token = jax.ffi.ffi_call(
        "allreduce",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), op_handle=op.py2f())
    return result, token


@partial(jax.custom_vjp, nondiff_argnames=["op", "comm"])
def allreduce(x: jax.Array, token: jax.Array, op, comm=None):
    if comm is None:
        comm = get_default_comm()

    # token = _token_manager.get_token()
    result, new_token = _allreduce_impl(x, token, comm, op)
    # _token_manager.update_token(new_token)
    return result, new_token


def allreduce_fwd(x, token, op, comm=None):
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allreduce_impl(x, token, comm, op)
    # residual should be things needed by backward; keep small
    return (result, new_token), (op,)


def allreduce_bwd(op, res, g):
    # implement gradient logic when you need it
    raise NotImplementedError("Backward pass for allreduce is not implemented yet.")
    # if op == MPI.SUM:  # MPI_SUM
    #     # For sum, gradient is broadcast from root to all ranks
    #     return (_bcast_impl(g, 0),)
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
    # else:
    #     raise ValueError(f"Unsupported op: {op}")


allreduce.defvjp(allreduce_fwd, allreduce_bwd)
