import pytest
from mpi4py import MPI

from jax import grad, jit
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi import gen_token
from jpi import gather

from tests.testing_utils import generate_array


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_gather(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Each rank creates different data
    arr = generate_array(array_shape, dtype) + rank

    token = gen_token()
    y, _ = gather(arr, token, root=0, comm=comm)

    # Only root receives the gathered data
    if rank == 0:
        sendcount = arr.shape[0]
        for r in range(size):
            start = r * sendcount
            end = start + sendcount
            expected = generate_array(array_shape, dtype) + r
            assert jnp.allclose(y[start:end], expected)
    else:
        # On non-root ranks, y may be zeros or uninitialized, so we skip the check
        pass


@pytest.mark.mpi(min_size=2)
def test_gather_jit(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Each rank creates different data
    arr = generate_array(array_shape, dtype) + rank

    def gather_fn(x):
        token = gen_token()
        y, _ = gather(arr, token, root=0, comm=comm)
        return y

    gather_jit = jit(gather_fn)

    y = gather_jit(arr)

    # Only root receives the gathered data
    if rank == 0:
        sendcount = arr.shape[0]
        for r in range(size):
            start = r * sendcount
            end = start + sendcount
            expected = generate_array(array_shape, dtype) + r
            assert jnp.allclose(y[start:end], expected)
    else:
        # On non-root ranks, y may be zeros or uninitialized, so we skip the check
        pass


@pytest.mark.mpi(min_size=2)
def test_gather_grad(
    array_shape: tuple,
    dtype: DTypeLike,
):
    arr = generate_array(array_shape, dtype) + rank

    def func(x):
        token = gen_token()
        y, _ = gather(x, token, root=0, comm=comm)
        # For testing, sum the gathered result to get a scalar output
        return jnp.sum(y)

    grad_fn = grad(func)
    grad_x = grad_fn(arr)

    # The gradient should be ones for all elements that contributed to the sum at the root,
    # and zeros elsewhere (for non-root ranks).
    expected = jnp.ones_like(arr)
    assert jnp.allclose(grad_x, expected)
