# test_gather_consistency.py
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
from jpi import gather, gen_token
import mpi4jax


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---- function that uses your gather (this is the code-path you want to test) ----
def loss(x):
    token = gen_token()
    gathered, _ = gather(x, token, root=0, comm=comm)
    return jnp.sum(jnp.sin(gathered))


def loss_with_mpi4jax(x):
    gathered = mpi4jax.gather(x, root=0, comm=comm)
    return jnp.sum(jnp.sin(gathered))


# Each rank contributes its own data
x = jnp.arange(0, 10, dtype=jnp.float32) + rank * 4

# ---- compute forward values and gradients ----
val = loss(x.copy())
val_mpi4jax = loss_with_mpi4jax(x.copy())


# ---- compare ----
assert np.allclose(val, val_mpi4jax), f"Rank {rank}: Forward values do not match!"
