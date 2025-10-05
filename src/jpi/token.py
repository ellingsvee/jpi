# Internal token for ordering MPI operations
from typing import TypeAlias
import jax.numpy as jnp
import jax

Token: TypeAlias = jax.Array


def gen_token() -> Token:
    """Generate a synchronization token for MPI operations.

    Returns:
        A scalar JAX array that serves as a synchronization token.

    Note:
        Tokens should be threaded through MPI operations to maintain proper ordering. Each MPI operation consumes a token and produces a new one.

    Example:
        ```python
        from jpi import bcast, allreduce, gen_token

        # Create initial token
        token = gen_token()

        # Thread token through operations
        result1, token = bcast(data1, token, root=0)
        result2, token = allreduce(data2, token, MPI.SUM)
        ```
    """
    return jnp.zeros((), dtype=jnp.float32)
