import pytest
import jax
import jax.numpy as jnp
from jpi.interface.allreduce import allreduce
from jpi.mpi import rank


def input_array():
    return jnp.arange(5, dtype=jnp.float32)


if __name__ == "__main__":
    x = input_array()
    print(f"- Rank {rank} before reduce: x = {x}")
    y = allreduce(x, op=0)  # op=0 is MPI.SUM
    print(f"- Rank {rank} after reduce: y = {y}")

    # grad_fn = jax.grad(lambda x: jnp.sum(allreduce(x, op=0)))
    # grad = grad_fn(input_array())
    # print(f"- Rank {rank} after reduce_grad: grad = {grad}")
