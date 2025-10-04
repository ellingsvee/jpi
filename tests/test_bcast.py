import pytest

from jax import grad, jit
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi.interface.token import gen_token
from jpi.interface.bcast import bcast

from testing_utils import generate_array

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_bcast(
    array_shape: tuple,
    dtype: DTypeLike,
):
    root = 0
    arr = generate_array(array_shape, dtype)

    if rank == root:
        x = arr
    else:
        x = jnp.empty_like(arr)

    token = gen_token()
    y, _ = bcast(x, token, root, comm=comm)

    if rank != root:
        assert jnp.allclose(arr, y)


@pytest.mark.mpi(min_size=2)
def test_bcast_jit(
    array_shape: tuple,
    dtype: DTypeLike,
):
    root = 0
    arr = generate_array(array_shape, dtype)

    def bcast_fn(x):
        token = gen_token()
        y, _ = bcast(x, token, root, comm=comm)
        return y

    bcast_jit = jit(bcast_fn)

    if rank == root:
        x = arr
    else:
        x = jnp.empty_like(arr)

    y = bcast_jit(x)

    if rank != root:
        assert jnp.allclose(arr, y)


@pytest.mark.mpi(min_size=2)
def test_bcast_grad(
    array_shape: tuple,
    dtype: DTypeLike,
):
    root = 0
    arr = generate_array(array_shape, dtype)

    def func(x):
        token = gen_token()
        y, _ = bcast(x, token, root, comm=comm)
        # For testing, sum the result to get a scalar output
        return jnp.sum(y)

    if rank == root:
        x = arr
    else:
        x = jnp.empty_like(arr)

    grad_fn = grad(func)
    grad_x = grad_fn(x)

    # At the root, the gradient should be all ones (since sum(y) w.r.t. x is 1)
    # At other ranks, the input is not used, so gradient should be zeros
    expected = jnp.ones_like(x) if rank == root else jnp.zeros_like(x)
    assert jnp.allclose(grad_x, expected)
