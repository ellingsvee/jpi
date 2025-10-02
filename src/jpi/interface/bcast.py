from functools import partial
import jax
import jax.numpy as jnp

from jpi.mpi import rank, size
from jpi.comm import get_default_comm
from jpi.interface.token import _token_manager


def _bcast_impl(x: jax.Array, token: jax.Array, comm, root):
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0, 1: 1}  # alias input and output buffers

    # NOTE: The root is unused in bcast. Just use the default root=0.
    result, token = jax.ffi.ffi_call(
        "bcast",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), root=root)
    return result, token


@partial(jax.custom_vjp, nondiff_argnames=["comm", "root"])
def bcast(x: jax.Array, root, comm=None):
    if comm is None:
        comm = get_default_comm()

    token = _token_manager.get_token()
    result, new_token = _bcast_impl(x, token, comm, root)
    _token_manager.update_token(new_token)

    return result


def bcast_fwd(x: jax.Array, root, comm=None):
    if comm is None:
        comm = get_default_comm()

    token = _token_manager.get_token()
    result, new_token = _bcast_impl(x, token, comm, root)
    _token_manager.update_token(new_token)

    return result, None


def bcast_bwd(root, comm, res: tuple, g: jax.Array):
    # raise NotImplementedError("The backward pass of bcast is not implemented.")
    return (bcast(g, root=root, comm=comm),)


bcast.defvjp(bcast_fwd, bcast_bwd)
