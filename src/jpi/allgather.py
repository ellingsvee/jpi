from functools import partial
import jax

from jpi.comm import get_default_comm, Comm
from jpi.token import Token


def _allgather_impl(x: jax.Array, token: Token, comm: Comm) -> tuple[jax.Array, Token]:
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
        "allgather",  # name of registered FFI function
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), numel_per_rank=numel)

    # result has shape (size,)+x.shape; return it and token_out
    return result, token_out


@partial(jax.custom_vjp, nondiff_argnames=["comm"])
def allgather(
    x: jax.Array, token: Token, comm: Comm | None = None
) -> tuple[jax.Array, Token]:
    """Gather arrays from all processes and distribute to all.

    Args:
        x: Local array to contribute to the gather operation. Must have the same shape on all processes except possibly the first dimension.
        token: Synchronization token for ordering operations.
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        result: Concatenated array with shape (total_elements, *x.shape[1:]), where total_elements = sum of x.shape[0] across all processes.
        new_token: Updated synchronization token.

    Example:
        ```python
        import jax.numpy as jnp
        from jpi import allgather
        from jpi import gen_token

        # Each rank contributes different data
        local_data = jnp.array([rank, rank + 1])  # rank-specific data
        token = gen_token()
        result, token = allgather(
            local_data, token
        )  # result contains data from all ranks concatenated
        ```
    """
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allgather_impl(x, token, comm)
    return result, new_token


def allgather_fwd(
    x: jax.Array, token: Token, comm: Comm | None = None
) -> tuple[tuple[jax.Array, Token], tuple[int, tuple]]:
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allgather_impl(x, token, comm)
    return (result, new_token), (x.shape[0], x.shape)


def allgather_bwd(
    comm: Comm, _: tuple, g: tuple[jax.Array, Token]
) -> tuple[jax.Array, Token]:
    # Gradient is simply the slice of g corresponding to this rank
    g_result, g_token = g

    rank = comm.Get_rank()
    return (g_result[rank], g_token)


allgather.defvjp(allgather_fwd, allgather_bwd)
