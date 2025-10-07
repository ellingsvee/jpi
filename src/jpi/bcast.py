from functools import partial
import jax
import jax.numpy as jnp

from jpi.comm import get_default_comm, Comm
from jpi.token import Token


def _bcast_impl(
    x: jax.Array, token: Token, comm: Comm, root: int
) -> tuple[jax.Array, Token]:
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0, 1: 1}  # alias input and output buffers

    result, token = jax.ffi.ffi_call(
        "bcast",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
        has_side_effect=True,
    )(x, token, comm_handle=comm.py2f(), root=root)
    return result, token


@partial(jax.custom_vjp, nondiff_argnames=["comm", "root"])
def bcast(
    x: jax.Array, token: Token, root: int, comm: Comm | None = None
) -> tuple[jax.Array, Token]:
    """Broadcast an array from one process to all others.

    Args:
        x: Input array to broadcast. Only meaningful on the root process.
        token: Synchronization token for ordering operations.
        root: Rank of the root process that owns the data to broadcast.
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        result: The broadcasted array (same on all processes).
        new_token: Updated synchronization token.

    Example:
        ```python
        import jax.numpy as jnp
        from jpi import bcast, gen_token

        # On rank 0: broadcast this data
        if rank == 0:
            data = jnp.array([1.0, 2.0, 3.0])
        else:
            data = jnp.zeros(3)  # Will be overwritten

        token = gen_token()
        result, token = bcast(
            data, token, root=0
        )  # Now all processes have [1.0, 2.0, 3.0]
        ```
    """
    if comm is None:
        comm = get_default_comm()

    result, new_token = _bcast_impl(x, token, comm, root)
    return result, new_token


def bcast_fwd(
    x: jax.Array, token: Token, root: int, comm: Comm | None = None
) -> tuple[tuple[jax.Array, Token], None]:
    if comm is None:
        comm = get_default_comm()

    result, new_token = _bcast_impl(x, token, comm, root)
    return (result, new_token), None


def bcast_bwd(root: int, comm: Comm, _, g: Token) -> tuple[jax.Array, Token]:
    g_result, token = g
    # Get rank from comm
    rank = comm.Get_rank()
    # Only root receives the gradient, others get zeros
    grad_x = g_result if rank == root else jnp.zeros_like(g_result)
    return (grad_x, token)


bcast.defvjp(bcast_fwd, bcast_bwd)
