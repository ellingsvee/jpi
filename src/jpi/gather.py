from functools import partial
import jax

from jpi.comm import get_default_comm, Comm
from jpi.token import Token


def _gather_impl(
    x: jax.Array, token: Token, comm: Comm, root: int
) -> tuple[jax.Array, Token]:
    rank = comm.Get_rank()
    size = comm.Get_size()

    # REQUIRE: every rank must provide same x.shape
    # Determine per-rank element count
    numel = int(x.size)  # number of elements each rank sends
    # Output shape: (size, ...) where ... is local shape
    out_shape = (size,) + tuple(x.shape)

    # JAX FFI types
    y_type = jax.ShapeDtypeStruct(out_shape, x.dtype)
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)

    # We keep aliasing token (output token out is alias of input token)
    # Map output index 1 (token_out) to input index 1 (token)
    input_output_aliases = {1: 1}

    # Build and call FFI. Pass comm handle and numel as extra args.
    # Note: exact signature names/ordering for jax.ffi.ffi_call vary by your FFI registration.
    result, token_out = jax.ffi.ffi_call(
        "gather",  # name of registered FFI function
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), numel_per_rank=numel, root=root)

    # result has shape (size,)+x.shape; return it and token_out
    return result, token_out


@partial(jax.custom_vjp, nondiff_argnames=["comm", "root"])
def gather(
    x: jax.Array, token: Token, root: int, comm: Comm | None = None
) -> tuple[jax.Array, Token]:
    """Gather arrays from all processes.

    Args:
        x: Local array to contribute to the gather operation. Must have the same shape on all processes except possibly the first dimension.
        token: Synchronization token for ordering operations.
        root: Rank of the root process that will receive the gathered data.
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        result: Concatenated array with shape (total_elements, *x.shape[1:]), where total_elements = sum of x.shape[0] across all processes.
        new_token: Updated synchronization token.

    Example:
        ```python
        import jax.numpy as jnp
        from jpi import gather, gen_token

        # Each rank contributes different data
        local_data = jnp.array([rank, rank + 1])  # rank-specific data
        token = gen_token()
        result, token = gather(
            local_data, token, root=0
        )  # rank 0 contains data from all ranks concatenated
        ```
    """
    if comm is None:
        comm = get_default_comm()
    result, new_token = _gather_impl(x, token, comm, root)
    return result, new_token


def gather_fwd(
    x: jax.Array, token: Token, root: int, comm: Comm | None = None
) -> tuple[tuple[jax.Array, Token], None]:
    if comm is None:
        comm = get_default_comm()
    result, new_token = _gather_impl(x, token, comm, root)
    return (result, new_token), None


def gather_bwd(
    root: int, comm: Comm, _, g: tuple[jax.Array, Token]
) -> tuple[jax.Array, Token]:
    # Import scatter here to avoid circular import
    from jpi.scatter import scatter

    g_result, g_token = g

    # We need to scatter the relevant slices back to each process
    scattered, g_token_new = scatter(g_result, g_token, root, comm)
    return (scattered, g_token_new)


gather.defvjp(gather_fwd, gather_bwd)
