import pytest
import jax
import jax.numpy as jnp
from jpi.interface.bcast import bcast
from jpi.mpi import rank

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def input_array():
    if rank == 0:
        return jnp.arange(5, dtype=jnp.float32)
    else:
        return jnp.zeros(5, dtype=jnp.float32)


if __name__ == "__main__":
    x = input_array()
    print(f"- Rank {rank} before bcast: x = {x}")
    y = bcast(x, root=0)
    print(f"- Rank {rank} after bcast: y = {y}")

    grad_fn = jax.grad(lambda x: jnp.sum(bcast(x, root=0)))
    grad = grad_fn(input_array())
    print(f"- Rank {rank} after bcast_grad: grad = {grad}")
