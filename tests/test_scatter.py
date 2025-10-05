import pytest
from mpi4py import MPI

from jax import grad, jit
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi import gen_token
from jpi import scatter

from testing_utils import generate_array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_scatter(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Only root provides the full array, others provide dummy input
    if rank == 0:
        arr = generate_array((array_shape[0] * size, *array_shape[1:]), dtype)
    else:
        arr = jnp.empty((array_shape[0] * size, *array_shape[1:]), dtype=dtype)

    token = gen_token()
    y, _ = scatter(arr, token, root=0, comm=comm)

    # # Each rank should receive its slice
    expected = generate_array((array_shape[0] * size, *array_shape[1:]), dtype)[
        rank * array_shape[0] : (rank + 1) * array_shape[0]
    ]
    assert jnp.allclose(y, expected)


@pytest.mark.mpi(min_size=2)
def test_scatter_jit(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Only root provides the full array, others provide dummy input
    if rank == 0:
        arr = generate_array((array_shape[0] * size, *array_shape[1:]), dtype)
    else:
        arr = jnp.empty((array_shape[0] * size, *array_shape[1:]), dtype=dtype)

    def scatter_fn(x):
        token = gen_token()
        y, _ = scatter(arr, token, root=0, comm=comm)
        return y

    scatter_jit = jit(scatter_fn)
    y = scatter_jit(arr)

    # # Each rank should receive its slice
    expected = generate_array((array_shape[0] * size, *array_shape[1:]), dtype)[
        rank * array_shape[0] : (rank + 1) * array_shape[0]
    ]
    assert jnp.allclose(y, expected)


@pytest.mark.mpi(min_size=2)
def test_scatter_grad(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Only root provides the full array, others provide dummy input
    if rank == 0:
        arr = generate_array((array_shape[0] * size, *array_shape[1:]), dtype)
    else:
        arr = jnp.empty((array_shape[0] * size, *array_shape[1:]), dtype=dtype)

    def func(x):
        token = gen_token()
        y, _ = scatter(x, token, root=0, comm=comm)
        return jnp.sum(y)

    grad_fn = grad(func)
    grad_x = grad_fn(arr)

    # Only the slice for this rank should have ones, rest should be zeros
    expected = jnp.zeros_like(arr)
    if rank == 0:
        expected = expected.at[:].set(1)

    assert jnp.allclose(grad_x, expected)
