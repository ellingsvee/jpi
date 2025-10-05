# test_bcast_consistency.py
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
from jpi import bcast, gen_token
import mpi4jax


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---- function that uses your bcast (this is the code-path you want to test) ----
def loss(x):
    token = gen_token()
    broadcasted, _ = bcast(x, token, root=0, comm=comm)
    return jnp.sum(jnp.sin(broadcasted))


def loss_with_mpi4jax(x):
    broadcasted = mpi4jax.bcast(x, root=0, comm=comm)
    return jnp.sum(jnp.sin(broadcasted))


# Only root needs meaningful data, others can have dummy data
if rank == 0:
    x = jnp.arange(1, 5, dtype=jnp.float32)
else:
    x = jnp.zeros(4, dtype=jnp.float32)

# ---- compute forward values and gradients ----
val = loss(x.copy())
val_mpi4jax = loss_with_mpi4jax(x.copy())

# grad = jax.grad(loss)(x.copy())
# grad_mpi4jax = jax.grad(loss_with_mpi4jax)(x.copy())


# ---- compare ----
assert np.allclose(val, val_mpi4jax), f"Rank {rank}: Forward values do not match!"
