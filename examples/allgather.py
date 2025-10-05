# test_allgather_consistency.py
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
from jpi import allgather, gen_token
import mpi4jax


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---- function that uses your allgather (this is the code-path you want to test) ----
def loss(x):
    token = gen_token()
    gathered, _ = allgather(x, token, comm=comm)
    print(f"Rank {rank} gathered data:\n{gathered}\n")
    return jnp.sum(jnp.sin(gathered))


def loss_with_mpi4jax(x):
    gathered = mpi4jax.allgather(x, comm=comm)
    # print(f"Rank {rank} gathered data with mpi4jax:\n{gathered}\n")
    return jnp.sum(jnp.sin(gathered))


# Each rank contributes its own data
x = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32) + rank * 4

# ---- compute forward values and gradients ----
val = loss(x.copy())
val_mpi4jax = loss_with_mpi4jax(x.copy())

# ---- compare ----
assert np.allclose(val, val_mpi4jax), f"Rank {rank}: Forward values do not match!"
