import time

import jax
import jax.numpy as jnp

from mpi4py import MPI
from jpi import gen_token, scatter, gather, barrier

from functools import partial

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 100
# Matrix of shape (size, n, n) filled with random numbers
if rank == 0:
    full = jnp.array(jax.random.normal(jax.random.PRNGKey(0), (size, n, n)))
    full = jnp.matmul(full, jnp.transpose(full, (0, 2, 1))) + n * jnp.eye(
        n
    )  # Make it positive definite
else:
    full = jnp.empty((size, n, n))

token = gen_token()


@jax.jit
def loss_no_mpi(x):
    # Take the logdet of each of the (n, n) matrices and sum them up
    return jnp.sum(jnp.linalg.slogdet(x)[1])


@jax.jit
def loss_with_mpi(x):
    token = gen_token()
    scattered, token = scatter(x, token, root=0, comm=comm)

    logdet_local = jnp.sum(jnp.linalg.slogdet(scattered)[1])
    logdet_local = jnp.expand_dims(logdet_local, 0)

    gathered, _ = gather(logdet_local, token, root=0, comm=comm)

    token = barrier(token, comm=comm)
    return jnp.sum(gathered)


if __name__ == "__main__":
    if rank == 0:
        _ = loss_no_mpi(full)  # Warm-up JIT

        start_val = time.time()
        loss_no_mpi_val = loss_no_mpi(full)
        end_val = time.time()
        print(f"Time taken without MPI: {end_val - start_val} seconds")

        loss_no_mpi_grad = jax.grad(loss_no_mpi)  # Warm-up JIT
        _ = loss_no_mpi_grad(full)
        start_grad = time.time()
        loss_no_mpi_grad = loss_no_mpi_grad(full)
        end_grad = time.time()
        print(f"Time taken for gradient without MPI: {end_grad - start_grad} seconds")
    else:
        loss_no_mpi_val = None
        loss_no_mpi_grad = None

    token = gen_token()

    _ = loss_with_mpi(full)  # Warm-up JIT
    start_val_mpi = time.time()
    loss_with_mpi_val = loss_with_mpi(full)
    end_val_mpi = time.time()

    token = barrier(token, comm=comm)

    loss_with_mpi_grad = jax.grad(loss_with_mpi)
    _ = loss_with_mpi_grad(full)  # Warm-up JIT
    start_grad_mpi = time.time()
    loss_with_mpi_grad = jax.grad(loss_with_mpi)(full)
    end_grad_mpi = time.time()

    token = barrier(token, comm=comm)

    if rank == 0:
        print(
            f"Rank {rank}: Time taken with MPI: {end_val_mpi - start_val_mpi} seconds"
        )
        print(
            f"Rank {rank}: Time taken for gradient with MPI: {end_grad_mpi - start_grad_mpi} seconds"
        )
    else:
        print(f"Rank {rank}: Completed MPI computations.")

    token = barrier(token, comm=comm)
    # comm.Barrier()
