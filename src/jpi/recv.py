from functools import partial
import jax

from jpi.comm import get_default_comm, Comm
from jpi.token import Token


def _recv_impl(
    x: jax.Array, token: Token, comm: Comm, source: int, tag: int = 0
) -> tuple[jax.Array, Token]:
    # Determine per-rank element count
    numel = int(x.size)  # number of elements each rank sends

    # JAX FFI types
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)

    # We keep aliasing token (output token out is alias of input token)
    # Alias input 0 (x) with output 0 (y)
    # Map output index 1 (token_out) to input index 1 (token)
    input_output_aliases = {0: 0, 1: 1}

    # Build and call FFI. Pass comm handle and numel as extra args.
    # Note: exact signature names/ordering for jax.ffi.ffi_call vary by your FFI registration.
    result, token_out = jax.ffi.ffi_call(
        "recv",  # name of registered FFI function
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), numel_per_rank=numel, source=source, tag=tag)

    return result, token_out


@partial(jax.custom_vjp, nondiff_argnames=["comm", "source", "tag"])
def recv(
    x: jax.Array,
    token: Token,
    source: int,
    tag: int = 0,
    comm: Comm | None = None,
) -> tuple[jax.Array, Token]:
    """Receive arrays

    Args:
        x: The array to receive data into. Only used to determine shape and dtype.
        token: Synchronization token for ordering operations.
        source: Source rank to receive data from.
        tag: Message tag for MPI communication.
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        result: The received array.
        new_token: Updated synchronization token.

    Example:
        ```python
        ```
    """
    if comm is None:
        comm = get_default_comm()
    result, new_token = _recv_impl(x, token, comm, source, tag)
    return result, new_token


def recv_fwd(
    x: jax.Array, token: Token, source: int, tag: int, comm: Comm | None = None
) -> tuple[tuple[jax.Array, Token], None]:
    if comm is None:
        comm = get_default_comm()
    result, new_token = _recv_impl(x, token, comm, source, tag)
    return (result, new_token), None


def recv_bwd(
    source: int, tag: int, comm: Comm, _, g: tuple[jax.Array, Token]
) -> tuple[jax.Array, Token]:
    # Import send here to avoid circular import
    from jpi.send import send

    g_result, g_token = g

    # We need to scatter the relevant slices back to each process
    g_result, g_token_new = send(g_result, g_token, dest=source, tag=tag, comm=comm)
    return (g_result, g_token_new)


recv.defvjp(recv_fwd, recv_bwd)
