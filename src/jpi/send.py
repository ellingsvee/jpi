from functools import partial
import jax

from jpi.comm import get_default_comm, Comm
from jpi.token import Token


def _send_impl(
    x: jax.Array, token: Token, comm: Comm, dest: int, tag: int = 0
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
        "send",  # name of registered FFI function
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, token, comm_handle=comm.py2f(), numel_per_rank=numel, dest=dest, tag=tag)

    return result, token_out


@partial(jax.custom_vjp, nondiff_argnames=["comm", "dest", "tag"])
def send(
    x: jax.Array, token: Token, dest: int, tag: int = 0, comm: Comm | None = None,
) -> tuple[jax.Array, Token]:
    """Send arrays

    Args:
        x: Local array to contribute to the send operation. 
        token: Synchronization token for ordering operations.
        dest: Destination rank to send data to.
        tag: Message tag for MPI communication.
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        result: The same array as x.
        new_token: Updated synchronization token.

    Example:
        ```python
        ```
    """
    if comm is None:
        comm = get_default_comm()
    result, new_token = _send_impl(x, token, comm, dest, tag)
    return result, new_token


def send_fwd(
    x: jax.Array, token: Token, dest: int, tag: int, comm: Comm | None = None
) -> tuple[tuple[jax.Array, Token], None]:
    if comm is None:
        comm = get_default_comm()
    result, new_token = _send_impl(x, token, comm, dest, tag)
    return (result, new_token), None


def send_bwd(dest: int, tag: int, comm: Comm, _, g: tuple[jax.Array, Token]) -> tuple[jax.Array, Token]:
    # Import recv here to avoid circular import
    from jpi.recv import recv
    
    g_result, g_token = g
    
    # We need to scatter the relevant slices back to each process
    g_result, g_token_new = recv(g_result, g_token, source=dest, tag=tag, comm=comm)
    return (g_result, g_token_new)


send.defvjp(send_fwd, send_bwd)
