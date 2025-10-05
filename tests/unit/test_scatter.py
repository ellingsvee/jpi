import pytest
from mpi4py import MPI

from jax import grad, jit
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi import gen_token
from jpi import scatter

from tests.testing_utils import generate_array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_scatter(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Have to do a separate implementation for when array_shape is 1D
    if len(array_shape) == 1:
        # Root provides full array of shape (size * array_shape[0],), others provide dummy input
        if rank == 0:
            arr = generate_array((size * array_shape[0],), dtype)
        else:
            arr = jnp.empty((size * array_shape[0],), dtype=dtype)
    else:
        # Root provides full array of shape (size, *array_shape), others provide dummy input
        if rank == 0:
            arr = generate_array((size, *array_shape), dtype)
        else:
            arr = jnp.empty((size, *array_shape), dtype=dtype)

    token = gen_token()
    y, _ = scatter(arr, token, root=0, comm=comm)

    # Again, separate implementation for 1D case
    if len(array_shape) == 1:
        expected = generate_array((size * array_shape[0],), dtype)[
            rank * array_shape[0] : (rank + 1) * array_shape[0]
        ]
    else:
        expected = generate_array((size, *array_shape), dtype)[rank]
    assert jnp.allclose(y, expected)


@pytest.mark.mpi(min_size=2)
def test_scatter_jit(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Have to do a separate implementation for when array_shape is 1D
    if len(array_shape) == 1:
        # Root provides full array of shape (size * array_shape[0],), others provide dummy input
        if rank == 0:
            arr = generate_array((size * array_shape[0],), dtype)
        else:
            arr = jnp.empty((size * array_shape[0],), dtype=dtype)
    else:
        # Root provides full array of shape (size, *array_shape), others provide dummy input
        if rank == 0:
            arr = generate_array((size, *array_shape), dtype)
        else:
            arr = jnp.empty((size, *array_shape), dtype=dtype)

    def scatter_fn(x):
        token = gen_token()
        y, _ = scatter(arr, token, root=0, comm=comm)
        return y

    scatter_jit = jit(scatter_fn)
    y = scatter_jit(arr)

    # Again, separate implementation for 1D case
    if len(array_shape) == 1:
        expected = generate_array((size * array_shape[0],), dtype)[
            rank * array_shape[0] : (rank + 1) * array_shape[0]
        ]
    else:
        expected = generate_array((size, *array_shape), dtype)[rank]
    assert jnp.allclose(y, expected)


# @pytest.mark.mpi(min_size=2)
# def test_scatter_grad(
#     array_shape: tuple,
#     dtype: DTypeLike,
# ):
#     # Only root provides the full array, others provide dummy input
#     if rank == 0:
#         arr = generate_array((array_shape[0] * size, *array_shape[1:]), dtype)
#     else:
#         arr = jnp.empty((array_shape[0] * size, *array_shape[1:]), dtype=dtype)

#     def func(x):
#         token = gen_token()
#         y, _ = scatter(x, token, root=0, comm=comm)
#         return jnp.sum(y)

#     grad_fn = grad(func)
#     grad_x = grad_fn(arr)

#     # Only the slice for this rank should have ones, rest should be zeros
#     expected = jnp.zeros_like(arr)
#     if rank == 0:
#         expected = expected.at[:].set(1)

#     assert jnp.allclose(grad_x, expected)


@pytest.mark.mpi(min_size=2)
def test_scatter_grad(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Only root provides the full array, others provide dummy input
    if len(array_shape) == 1:
        full_shape = (size * array_shape[0],)
    else:
        full_shape = (size,) + array_shape

    if rank == 0:
        arr = generate_array(full_shape, dtype)
    else:
        arr = jnp.empty(full_shape, dtype=dtype)

    def func(x):
        token = gen_token()
        y, _ = scatter(x, token, root=0, comm=comm)
        return jnp.sum(y)

    grad_fn = grad(func)
    grad_x = grad_fn(arr)

    # Build expected gradient: ones for this rank's slice, zeros elsewhere
    if rank == 0:
        expected = jnp.ones_like(arr)
    else:
        expected = jnp.zeros_like(arr)
    assert jnp.allclose(grad_x, expected)
