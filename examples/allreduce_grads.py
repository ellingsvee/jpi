# test_allreduce_grad.py
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
from jpi import allreduce, gen_token
import mpi4jax


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
op = MPI.SUM


# ---- function that uses your allreduce (this is the code-path you want to test) ----
def loss(x):
    token = gen_token()
    summed, _ = allreduce(x, token, op, comm=comm)
    return jnp.sum(jnp.sin(summed))


def loss_with_mpi4jax(x):
    summed = mpi4jax.allreduce(x, op)
    return jnp.sum(jnp.sin(summed))


x = jnp.arange(1, 5, dtype=jnp.float32) + rank

# ---- compute forward values and gradients ----
val = loss(x.copy())
val_mpi4jax = loss_with_mpi4jax(x.copy())

grad = jax.grad(loss)(x.copy())
grad_mpi4jax = jax.grad(loss_with_mpi4jax)(x.copy())


# ---- compare ----
if rank == 0:
    # print(f"  Forward values: {val} vs {val_mpi4jax}")
    # print(f"  Gradients: {grad} vs {grad_mpi4jax}")
    print("Forward values:")
    print(f" - jpi: {val}")
    print(f" - mpi4jax: {val_mpi4jax}")
    assert np.allclose(val, val_mpi4jax), "Forward values do not match!"

    print("Gradients:")
    print(f" - jpi: {grad}")
    print(f" - mpi4jax: {grad_mpi4jax}")
    assert np.allclose(grad, grad_mpi4jax), "Gradients do not match!"
