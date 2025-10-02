import jax
import jax.numpy as jnp
from jpi.interface.allgather import allgather
# from jpi.mpi import rank

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def input_array():
    return jnp.arange(2, dtype=jnp.float32) + rank * 2


@jax.jit
def test(x):
    y = allgather(x, comm=comm)
    return y


if __name__ == "__main__":
    x = input_array()
    print(f"- Rank {rank} before allgather: x = {x}")
    # y = allgather(x)
    y = test(x)
    print(f"- Rank {rank} after alltgather: y = {y}")

MPI.SUM
