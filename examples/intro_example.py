import jax
import jax.numpy as jnp
from mpi4py import MPI
from jpi import allreduce, scatter, gen_token

# Use mpi4py communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Fill the root rank with data, others with zeros
if rank == 0:
    x = jnp.arange(2 * size, dtype=jnp.float32)
else:
    x = jnp.zeros(2 * size, dtype=jnp.float32)


# Some example function that uses scatter and allreduce
def func(x):
    # Generate token for synchronization between operations
    token = gen_token()

    # Scatter x from rank 0 to all ranks
    x, token = scatter(x, token, root=0, comm=comm)

    # Each rank can do something different with the array
    if rank == size - 1:
        x = x * 2

    # Perform allreduce (sum) on the scattered array
    result, token = allreduce(x, token, op=MPI.SUM, comm=comm)
    return jnp.sum(result)


# JIT and grad the function
func_jit = jax.jit(func)
func_grad = jax.grad(func)

# Compute result and gradient
result = func_jit(x)
grad_result = func_grad(x)
print(f"Rank {comm.rank} has result {result} and gradient {grad_result}")

# Out
# Rank 0 has result 41.0 and gradient [1. 1. 1. 1. 1. 1. 2. 2.]
# Rank 1 has result 41.0 and gradient [0. 0. 0. 0. 0. 0. 0. 0.]
# Rank 2 has result 41.0 and gradient [0. 0. 0. 0. 0. 0. 0. 0.]
# Rank 3 has result 41.0 and gradient [0. 0. 0. 0. 0. 0. 0. 0.]
