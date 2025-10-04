import jax

from jpi.comm import get_default_comm

from functools import partial


def _barrier_impl(token: jax.Array, comm):
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0}  # alias input and output buffers

    token = jax.ffi.ffi_call(
        "barrier",
        (token_type,),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(token, comm_handle=comm.py2f())[0]
    return token


@partial(jax.custom_vjp, nondiff_argnames=["comm"])
def barrier(token: jax.Array, comm=None):
    if comm is None:
        comm = get_default_comm()
    new_token = _barrier_impl(token, comm)
    return new_token


def barrier_fwd(token: jax.Array, comm=None):
    if comm is None:
        comm = get_default_comm()
    new_token = _barrier_impl(token, comm)
    return new_token, None


def barrier_bwd(comm, _, g):
    return (g,)


barrier.defvjp(barrier_fwd, barrier_bwd)
