from functools import partial
import jax
import jax.numpy as jnp

from jpi.mpi import rank, size
from jpi.comm import get_default_comm
from jpi.interface.token import _token_manager


def _barrier_impl(token: jax.Array, comm):
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0}  # alias input and output buffers

    # NOTE: The root is unused in barrier. Just use the default root=0.
    token = jax.ffi.ffi_call(
        "barrier",
        (token_type,),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(token, comm_handle=comm.py2f())[0]
    return token


@partial(jax.custom_vjp, nondiff_argnames=["comm"])
def barrier(comm=None):
    if comm is None:
        comm = get_default_comm()

    token = _token_manager.get_token()
    new_token = _barrier_impl(token, comm)
    _token_manager.update_token(new_token)

    return None


def barrier_fwd(comm=None):
    if comm is None:
        comm = get_default_comm()

    token = _token_manager.get_token()
    result, new_token = _barrier_impl(token, comm)
    _token_manager.update_token(new_token)

    return result, None


def barrier_bwd(comm, root, res: tuple, g: jax.Array):
    raise NotImplementedError("The backward pass of barrier is not implemented.")


barrier.defvjp(barrier_fwd, barrier_bwd)
