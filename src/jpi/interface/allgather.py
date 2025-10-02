from functools import partial
import jax
import jax.numpy as jnp

from jpi.mpi import rank, size
from jpi.comm import get_default_comm
from jpi.interface.token import _token_manager


def _allgather_impl(x: jax.Array, token: jax.Array, comm):
    # The input and output of the ffi_call must have the same shape and dtype
    # since we are aliasing them. For allgather, the output shape is (size * x.shape[0], ...).
    # Therefore we need to expand the input x.
    sendcount = x.shape[0]
    x_full = jnp.zeros(size * sendcount, dtype=x.dtype)
    x_full = x_full.at[0:sendcount].set(x)

    y_type = jax.ShapeDtypeStruct(x_full.shape, x_full.dtype)
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0, 1: 1}  # alias input and output buffers

    result, token = jax.ffi.ffi_call(
        "allgather",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x_full, token, comm_handle=comm.py2f(), sendcount=sendcount)
    return result, token


@partial(jax.custom_vjp, nondiff_argnames=["comm"])
def allgather(x: jax.Array, comm=None):
    if comm is None:
        comm = get_default_comm()

    token = _token_manager.get_token()
    result, new_token = _allgather_impl(x, token, comm)
    _token_manager.update_token(new_token)

    return result


def allgather_fwd(x: jax.Array, *, comm=None):
    if comm is None:
        comm = get_default_comm()

    token = _token_manager.get_token()
    result, new_token = _allgather_impl(x, token, comm)
    _token_manager.update_token(new_token)

    return result, (x.shape[0],)


def allgather_bwd(res: tuple, g: jax.Array):
    (sendcount,) = res
    # Gradient is simply the slice of g corresponding to this rank
    start = rank * sendcount
    end = start + sendcount
    return (g[start:end],)


allgather.defvjp(allgather_fwd, allgather_bwd)
