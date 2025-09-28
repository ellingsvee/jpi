import jax
import jax.numpy as jnp
from mpi4py import MPI
from jpi.interface.scatter import scatter

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Input: Meaningful only on root=0
if rank == 0:
    x = jnp.arange(size * 2, dtype=jnp.float32) * 1.0
else:
    x = jnp.zeros(size * 2, dtype=jnp.float32)  # Dummy


if __name__ == "__main__":
    print(f"Rank {rank} before scatter: x = {x}")
    y = scatter(x, root=0)
    print(f"Rank {rank} after scatter: y = {y}")
