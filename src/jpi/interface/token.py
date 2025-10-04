# Internal token for ordering MPI operations
import jax.numpy as jnp


def gen_token():
    """Generate a synchronization token for MPI operations.

    Returns:
        A scalar JAX array that serves as a synchronization token.

    Note:
        Tokens should be threaded through MPI operations to maintain proper ordering. Each MPI operation consumes a token and produces a new one.

    Example:
        ```python
        from jpi.interface.token import gen_token
        from jpi.interface import bcast, allreduce

        # Create initial token
        token = gen_token()

        # Thread token through operations
        result1, token = bcast(data1, token, root=0)
        result2, token = allreduce(data2, token, MPI.SUM)
        ```
    """
    return jnp.zeros((), dtype=jnp.float32)
