from functools import partial
import jax

from jpi.comm import get_default_comm


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
def bcast(x: jax.Array, token: jax.Array, root, comm=None):
    if comm is None:
        comm = get_default_comm()

    result, new_token = _bcast_impl(x, token, comm, root)
    return result, new_token


def bcast_fwd(x: jax.Array, token: jax.Array, root, comm=None):
    if comm is None:
        comm = get_default_comm()

    result, new_token = _bcast_impl(x, token, comm, root)
    return (result, new_token), None


def bcast_bwd(root, comm, res: tuple, g: tuple):
    # raise NotImplementedError("The backward pass of bcast is not implemented.")
    g, token = g
    result, new_token = bcast(g, token, root=root, comm=comm)
    return (result, new_token)


bcast.defvjp(bcast_fwd, bcast_bwd)
