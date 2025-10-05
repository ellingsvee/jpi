import jax
import jax.numpy as jnp

from mpi4py import MPI
from jpi import gen_token, scatter, gather

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 2
# Matrix of shape (size, n, n) filled with random numbers
if rank == 0:
    full = jnp.array(jax.random.normal(jax.random.PRNGKey(0), (size, n, n)))
    full = jnp.matmul(full, jnp.transpose(full, (0, 2, 1))) + n * jnp.eye(
        n
    )  # Make it positive definite
else:
    full = jnp.empty((size, n, n))

token = gen_token()


def loss_no_mpi(x):
    # Take the logdet of each of the (n, n) matrices and sum them up
    return jnp.sum(jnp.linalg.slogdet(x)[1])


def loss_with_mpi(x):
    token = gen_token()
    # global token

    scattered, token = scatter(x, token, root=0, comm=comm)

    logdet_local = jnp.sum(jnp.linalg.slogdet(scattered)[1])
    logdet_local = jnp.expand_dims(logdet_local, 0)

    gathered, _ = gather(logdet_local, token, root=0, comm=comm)
    return jnp.sum(gathered)


if __name__ == "__main__":
    if rank == 0:
        loss_no_mpi_val = loss_no_mpi(full)
        loss_no_mpi_grad = jax.grad(loss_no_mpi)(full)
        print(f"Rank {rank}: Loss without MPI: {loss_no_mpi_val}")
        print(f"Rank {rank}: Gradient without MPI: {loss_no_mpi_grad}")
    else:
        loss_no_mpi_val = None
        loss_no_mpi_grad = None

    loss_with_mpi_val = loss_with_mpi(full)
    loss_with_mpi_grad = jax.grad(loss_with_mpi)(full)
    if rank == 0:
        print(f"Rank {rank}: Loss with MPI: {loss_with_mpi_val}")
        print(f"Rank {rank}: Gradient with MPI: {loss_with_mpi_grad}")
