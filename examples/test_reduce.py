import jax
import jax.numpy as jnp
from mpi4py import MPI
from jpi.interface.reduce import reduce

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


MPI.SUM

x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)


def test_func(x: jax.Array):
    return reduce(x, root=0, op=0)


def test_func_2(x: jax.Array):
    return reduce(x, root=0, op=0).sum()


reduce_grad = jax.grad(test_func_2)

if __name__ == "__main__":
    print(f"Rank {rank} before reduce: {x}")
    y = test_func(jnp.copy(x))  # Ensure x is not donated
    if rank == 0:
        print(f"Rank {rank} after reduce: {y}")

    x_grad = reduce_grad(x)
    print(f"Rank {rank} after reduce_grad: {x_grad}")
