from functools import partial

from mpi4py import MPI

import jax
import jax.numpy as jnp
from jpi.comm import get_default_comm, Comm
from jpi.token import Token
from jpi.op import Op


def _allreduce_impl(
    x: jax.Array, token: Token, comm: Comm, op: Op
) -> tuple[jax.Array, Token]:
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)

    token_type = jax.ShapeDtypeStruct(token.shape, token.dtype)
    input_output_aliases = {1: 1}  # alias input and output buffers

    result, token = jax.ffi.ffi_call(
        "allreduce",
        (y_type, token_type),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
        has_side_effect=True,
    )(x, token, comm_handle=comm.py2f(), op_handle=op.py2f())
    return result, token


@partial(jax.custom_vjp, nondiff_argnames=["op", "comm"])
def allreduce(
    x: jax.Array, token: Token, op: Op, comm: Comm | None = None
) -> tuple[jax.Array, Token]:
    """Perform a reduction operation across all processes.

    Args:
        x: Local array to contribute to the reduction.
        token: Synchronization token for ordering operations.
        op: MPI reduction operation. Supported operations include:
            - MPI.SUM: Element-wise sum across all processes
            - MPI.PROD: Element-wise product across all processes
            - MPI.MAX: Element-wise maximum across all processes
            - MPI.MIN: Element-wise minimum across all processes
        comm: MPI communicator. If None, uses the default communicator.

    Returns:
        result: Array containing the reduction result (same on all processes).
        new_token: Updated synchronization token.

    Raises:
        NotImplementedError: If the backward pass is not implemented for the
            specified reduction operation.

    Example:
        ``` python
        import jax.numpy as jnp
        from jpi import allreduce, gen_token
        from mpi4py import MPI

        # Sum arrays across all processes
        local_data = jnp.array([1.0, 2.0]) * (rank + 1)
        token = gen_token()
        result, token = allreduce(
            local_data, token, MPI.SUM
        )  # result contains the sum from all processes
        ```
    """
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allreduce_impl(x, token, comm, op)
    return result, new_token


def allreduce_fwd(
    x: jax.Array, token: Token, op: Op, comm: Comm | None = None
) -> tuple[tuple[jax.Array, Token], tuple[jax.Array, jax.Array]]:
    if comm is None:
        comm = get_default_comm()
    result, new_token = _allreduce_impl(x, token, comm, op)
    return (result, new_token), (x, result)


def allreduce_bwd(op: Op, _, res: tuple, g: tuple) -> tuple[jax.Array, Token]:
    g_result, g_token = g  # gradients w.r.t. outputs
    x, y = res

    if op == MPI.SUM:
        grad_out = g_result
    elif op == MPI.PROD:
        grad_out = jnp.where(x != 0, y / x * g_result, 0)
    elif op == MPI.MAX:
        grad_out = jnp.where(x == y, g_result, 0)
    elif op == MPI.MIN:
        grad_out = jnp.where(x == y, g_result, 0)
    else:
        raise NotImplementedError("Backward for this op not implemented.")

    return (grad_out, g_token)


allreduce.defvjp(allreduce_fwd, allreduce_bwd)
