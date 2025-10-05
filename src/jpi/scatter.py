from functools import partial
import jax
import jax.numpy as jnp

from jpi.comm import get_default_comm, Comm
from jpi.token import Token


def _scatter_impl(x: jax.Array, token: Token, comm: Comm, root: int):
    size = comm.Get_size()

    if x.shape[0] % size != 0:
        raise ValueError(
            f"x.shape[0] ({x.shape[0]}) must be divisible by number of processes ({size})"
        )
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)

    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0, 1: 1}  # alias input and output buffers

    y_unsliced, token = jax.ffi.ffi_call(
        "scatter",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), root=root)
    return y_unsliced[: x.shape[0] // size], token


@partial(jax.custom_vjp, nondiff_argnames=["comm", "root"])
def scatter(
    x: jax.Array, token: Token, root: int, comm: Comm | None = None
) -> tuple[jax.Array, Token]:
    """Distribute arrays to all processes.

    Args:
        x: Local array to contribute to the scatter operation. Must have the same shape on all processes except possibly the first dimension.
        token: Synchronization token for ordering operations.
        root: Rank of the root process that will distribute the data.
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        result: Sliced array with shape (x.shape[0] // size, *x.shape[1:]), where size is the number of processes.
        new_token: Updated synchronization token.

    Example:
        ```python
        import jax.numpy as jnp
        from jpi import scatter, gen_token

        # Each rank contributes different data
        local_data = jnp.array([rank, rank + 1])  # rank-specific data
        token = gen_token()
        result, token = scatter(local_data, token, root=0)
        ```
    """
    if comm is None:
        comm = get_default_comm()
    result, new_token = _scatter_impl(x, token, comm, root)
    return result, new_token


def scatter_fwd(
    x: jax.Array, token: Token, root: int, comm: Comm | None = None
) -> tuple[tuple[jax.Array, Token], tuple[int]]:
    if comm is None:
        comm = get_default_comm()
    result, new_token = _scatter_impl(x, token, comm, root)
    return (result, new_token), None


def scatter_bwd(
    root: int, comm: Comm, res: tuple, g: jax.Array
) -> tuple[jax.Array, Token]:
    # Import gather here to avoid circular import
    from jpi.gather import gather

    g_result, g_token = g

    # Only root should receive the gathered gradients
    gathered, g_token_new = gather(g_result, g_token, root, comm)
    if comm.Get_rank() == root:
        return (gathered, g_token_new)
    else:
        # Non-root processes contribute their gradients but get zeros back
        zeros = jnp.zeros_like(gathered)  # This should be the shape of the full input
        return (zeros, g_token_new)


scatter.defvjp(scatter_fwd, scatter_bwd)
