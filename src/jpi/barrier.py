import jax

from jpi.comm import get_default_comm

from jpi.comm import Comm
from jpi.token import Token

from functools import partial


def _barrier_impl(token: Token, comm: Comm) -> Token:
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0}  # alias input and output buffers

    token = jax.ffi.ffi_call(
        "barrier",
        (token_type,),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
        has_side_effect=True,
    )(token, comm_handle=comm.py2f())[0]
    return token


@partial(jax.custom_vjp, nondiff_argnames=["comm"])
def barrier(token: Token, comm: Comm | None = None) -> Token:
    """Synchronize all processes in the communicator.

    Args:
        token: Synchronization token for ordering operations. The token is passed through unchanged but ensures proper sequencing.
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        new_token: Updated synchronization token (same value as input).

    Example:
        ```python
        import jax.numpy as jnp
        from jpi import barrier, gen_token

        # Ensure all processes reach this point before continuing
        token = gen_token()
        token = barrier(token)  # Now all processes have synchronized
        ```
    """
    if comm is None:
        comm = get_default_comm()
    new_token = _barrier_impl(token, comm)
    return new_token


def barrier_fwd(token: Token, comm: Comm | None = None) -> tuple[Token, None]:
    if comm is None:
        comm = get_default_comm()
    new_token = _barrier_impl(token, comm)
    return new_token, None


def barrier_bwd(comm: Comm, _, g: Token) -> tuple[Token]:
    return (g,)


barrier.defvjp(barrier_fwd, barrier_bwd)
