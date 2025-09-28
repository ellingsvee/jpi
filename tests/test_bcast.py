import pytest
import jax
import jax.numpy as jnp
from jpi.interface.bcast import bcast
from jpi.mpi import rank


def input_array():
    if rank == 0:
        return jnp.arange(5, dtype=jnp.float32)
    else:
        return jnp.zeros(5, dtype=jnp.float32)


@jax.jit
def function_to_be_jitted(x):
    return jnp.sum(bcast(x, root=0))


if __name__ == "__main__":
    x = input_array()
    print(f"- Rank {rank} before bcast: x = {x}")
    y = bcast(x, root=0)
    print(f"- Rank {rank} after bcast: y = {y}")

    x = input_array()
    print(f"- Rank {rank} before bcast_jit: x = {x}")
    y = function_to_be_jitted(x)
    print(f"- Rank {rank} after bcast_jit: y = {y}")

    grad_fn = jax.grad(lambda x: jnp.sum(bcast(x, root=0)))
    grad = grad_fn(input_array())
    print(f"- Rank {rank} after bcast_grad: grad = {grad}")
