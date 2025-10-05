import jax
import jax.numpy as jnp
from mpi4py import MPI
from jpi import allreduce, gen_token

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Each rank starts with its own data
data = jnp.array([comm.rank], dtype=jnp.float32)


# Some function that uses allreduce
def func(data):
    token = gen_token()
    result, _ = allreduce(data, token, op=MPI.SUM, comm=comm)

    # Each rank can do something different with the result
    if rank == 0:
        result = result * 2
    return jnp.sum(result)


# JIT and grad the function
func_jit = jax.jit(func)
func_grad = jax.grad(func_jit)

# Compute result and gradient
result = func_jit(data)
grad_result = func_grad(data)
print(f"Rank {comm.rank} has result {result} and gradient {grad_result}")
