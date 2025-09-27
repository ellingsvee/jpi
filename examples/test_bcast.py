import jax
import jax.numpy as jnp
from mpi4py import MPI
from jpi.interface.bcast import bcast

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Input: Meaningful only on root=0
if rank == 0:
    x = jnp.arange(5, dtype=jnp.float32) * 2.0  # [0, 2, 4, ..., 18]
else:
    x = jnp.zeros(5, dtype=jnp.float32)  # Dummy


def test_func(x):
    y = bcast(x, root=0)
    return jnp.sum(y)


bcast_grad = jax.grad(test_func)

if __name__ == "__main__":
    # print(f"Rank {rank} before bcast: x = {x}")
    # x_bcast = bcast(x, root=0)
    # print(f"Rank {rank} after bcast: x_bcast = {x_bcast}")
    x_bcast_grad = bcast_grad(x)
    print(f"Rank {rank} after bcast_grad: x_bcast_grad = {x_bcast_grad}")
