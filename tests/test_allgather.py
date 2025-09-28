import pytest
import jax.numpy as jnp
from jpi.interface.allgather import allgather
from jpi.mpi import rank


def input_array():
    return jnp.arange(2, dtype=jnp.float32) + rank * 2


if __name__ == "__main__":
    x = input_array()
    print(f"- Rank {rank} before allgather: x = {x}")
    y = allgather(x)
    print(f"- Rank {rank} after alltgather: y = {y}")
