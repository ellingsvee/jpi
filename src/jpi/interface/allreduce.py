from functools import partial

from mpi4py import MPI

import jax
import jax.numpy as jnp
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
    result, new_token = _allreduce_impl(x, token, comm, op)
    return result, new_token


def allreduce_fwd(x, token, op, comm=None):
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allreduce_impl(x, token, comm, op)
    return (result, new_token), (x, result)


def allreduce_bwd(op, comm, res, g):
    g_result, g_token = g  # gradients w.r.t. outputs
    x, y = res

    if op == MPI.SUM:
        grad_out, g_token = allreduce(g_result, g_token, op, comm=comm)
    elif op == MPI.PROD:
        grad_out = jnp.where(x != 0, y / x * g_result, 0)
    elif op == MPI.MAX:
        grad_out = jnp.where(x == y, g_result, 0)
    elif op == MPI.MIN:
        grad_out = jnp.where(x == y, g_result, 0)
    else:
        raise NotImplementedError("Backward for this op not implemented.")

    return (grad_out, g_token)


allreduce.defvjp(allreduce_fwd, allreduce_bwd)
