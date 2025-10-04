import pytest
from mpi4py import MPI

from jax import grad, jit
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi.interface.token import gen_token
from jpi.interface.allgather import allgather

from jpi.testing_utils import generate_array


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_allgather(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Each rank creates different data
    arr = generate_array(array_shape, dtype) + rank

    token = gen_token()
    y, _ = allgather(arr, token, comm=comm)

    # Check that the result contains data from all ranks
    sendcount = arr.shape[0]
    for r in range(size):
        start = r * sendcount
        end = start + sendcount
        expected = generate_array(array_shape, dtype) + r
        assert jnp.allclose(y[start:end], expected)


@pytest.mark.mpi(min_size=2)
def test_allgather_jit(
    array_shape: tuple,
    dtype: DTypeLike,
):
    arr = generate_array(array_shape, dtype) + rank

    def allgather_fn(x):
        token = gen_token()
        y, _ = allgather(x, token, comm=comm)
        return y

    allgather_jit = jit(allgather_fn)

    y = allgather_jit(arr)

    # Check that the result contains data from all ranks
    sendcount = arr.shape[0]
    for r in range(size):
        start = r * sendcount
        end = start + sendcount
        expected = generate_array(array_shape, dtype) + r
        assert jnp.allclose(y[start:end], expected)


@pytest.mark.mpi(min_size=2)
def test_allgather_grad(
    array_shape: tuple,
    dtype: DTypeLike,
):
    arr = generate_array(array_shape, dtype) + rank

    def func(x):
        token = gen_token()
        y, _ = allgather(x, token, comm=comm)
        # For testing, sum the result to get a scalar output
        return jnp.sum(y)

    grad_fn = grad(func)
    grad_x = grad_fn(arr)

    # For allgather, the gradient should be all ones since each element
    # contributes once to the sum of the gathered result
    expected = jnp.ones_like(arr)
    assert jnp.allclose(grad_x, expected)
