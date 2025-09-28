import pytest
import jax
import jax.numpy as jnp
from jpi.interface.scatter import scatter
from jpi.mpi import rank, size


def input_array():
    if rank == 0:
        return jnp.arange(2 * size, dtype=jnp.float32)
    else:
        return jnp.zeros(2 * size, dtype=jnp.float32)


if __name__ == "__main__":
    x = input_array()
    print(f"- Rank {rank} before scatter: x = {x}")
    y = scatter(x, root=0)
    print(f"- Rank {rank} after scatter: y = {y}")

    # grad_fn = jax.grad(lambda x: jnp.sum(scatter(x, root=0)))
    # grad = grad_fn(input_array())
    # print(f"- Rank {rank} after scatter_grad: grad = {grad}")
