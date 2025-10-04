from functools import partial
import jax
import jax.numpy as jnp

from jpi.comm import get_default_comm


def _allgather_impl(x: jax.Array, token: jax.Array, comm):
    # The input and output of the ffi_call must have the same shape and dtype
    # since we are aliasing them. For allgather, the output shape is (size * x.shape[0], ...).
    # Therefore we need to expand the input x.
    sendcount = x.shape[0]
    x_full = jnp.zeros(comm.Get_size() * sendcount, dtype=x.dtype)
    x_full = x_full.at[0:sendcount].set(x)

    y_type = jax.ShapeDtypeStruct(x_full.shape, x_full.dtype)
    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {0: 0, 1: 1}  # alias input and output buffers

    result, token = jax.ffi.ffi_call(
        "allgather",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x_full, token, comm_handle=comm.py2f(), sendcount=sendcount)
    return result, token


@partial(jax.custom_vjp, nondiff_argnames=["comm"])
def allgather(x: jax.Array, token: jax.Array, comm=None):
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
        from jpi.interface import allgather
        from jpi.interface.token import gen_token

        # Each rank contributes different data
        local_data = jnp.array([rank, rank + 1])  # rank-specific data
        token = gen_token()
        result, token = allgather(local_data, token) # result contains data from all ranks concatenated
        ```
    """
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allgather_impl(x, token, comm)
    return result, new_token


def allgather_fwd(x: jax.Array, token: jax.Array, comm=None):
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allgather_impl(x, token, comm)
    return (result, new_token), (x.shape[0],)


def allgather_bwd(comm, res: tuple, g: jax.Array):
    (sendcount,) = res

    # Gradient is simply the slice of g corresponding to this rank
    g_result, g_token = g

    rank = comm.Get_rank()
    start = rank * sendcount
    end = start + sendcount
    return (g_result[start:end], g_token)


allgather.defvjp(allgather_fwd, allgather_bwd)
