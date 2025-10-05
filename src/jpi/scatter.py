from functools import partial
import jax
import jax.numpy as jnp

from jpi.comm import get_default_comm, Comm
from jpi.token import Token


def _scatter_impl(x: jax.Array, token: Token, comm: Comm, root: int):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if x.shape[0] % size != 0:
        raise ValueError(
            f"x.shape[0] ({x.shape[0]}) must be divisible by number of processes ({size})"
        )

    # Each rank's output shape is 1/size slice along axis 0
    out_shape = (x.shape[0] // size,) + x.shape[1:]
    y_type = jax.ShapeDtypeStruct(out_shape, x.dtype)
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)

    input_output_aliases = {1: 1}

    numel = int(x.size)  # total number of elements (only relevant on root)

    y, token_out = jax.ffi.ffi_call(
        "scatter",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), root=root, numel_per_rank=numel)

    # Squeeze leading dimension if it's 1
    if y.shape[0] == 1:
        y = jnp.squeeze(y, axis=0)

    return y, token_out


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
) -> tuple[tuple[jax.Array, Token], tuple[int, ...]]:
    if comm is None:
        comm = get_default_comm()
    result, new_token = _scatter_impl(x, token, comm, root)
    return (result, new_token), x.shape


# def scatter_bwd(
#     root: int, comm: Comm, res: tuple, g: jax.Array
# ) -> tuple[jax.Array, Token]:
#     # Import gather here to avoid circular import
#     from jpi.gather import gather

#     g_result, g_token = g
#     x_shape = res

#     gathered, g_token_new = gather(g_result, g_token, root, comm)

#     # ...

#     return (gathered, g_token_new)


def scatter_bwd(root: int, comm: Comm, res: tuple, g: tuple) -> tuple[jax.Array, Token]:
    # Import gather here to avoid circular import
    from jpi.gather import gather

    # g is the cotangent for the primal outputs: (y_cotangent, token_cotangent)
    g_result, g_token = g
    x_shape = res  # x.shape saved as residual in scatter_fwd

    # gather will assemble the per-rank slices into the full x-shaped array on the root
    gathered, g_token_new = gather(g_result, g_token, root, comm)

    # Make sure non-root ranks return a zero array with the same shape as the primal x.
    rank = comm.Get_rank()
    if rank != root:
        # create zeros with the correct dtype and shape
        x_grad = jnp.zeros(x_shape, dtype=g_result.dtype)
    else:
        # On root we expect 'gathered' to already have the full shape.
        # Ensure it matches the saved x_shape (reshape if necessary).
        x_grad = jnp.asarray(gathered)
        if x_grad.shape != x_shape:
            x_grad = jnp.reshape(x_grad, x_shape)

    # Return cotangents in the same order as the primal inputs: (x, token)
    return (x_grad, g_token_new)


scatter.defvjp(scatter_fwd, scatter_bwd)
