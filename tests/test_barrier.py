import pytest
from mpi4py import MPI

from jax import jit, grad
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi import gen_token
from jpi import barrier


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_barrier(
    array_shape: tuple,
    dtype: DTypeLike,
):
    """Test that barrier synchronizes all processes"""
    token = gen_token()

    # Simple test - barrier should return without error
    new_token = barrier(token, comm=comm)

    # Token should have the same shape and dtype
    assert new_token.shape == token.shape
    assert new_token.dtype == token.dtype


@pytest.mark.mpi(min_size=2)
def test_barrier_jit(
    array_shape: tuple,
    dtype: DTypeLike,
):
    """Test that barrier works under JIT compilation"""

    def barrier_fn(token):
        return barrier(token, comm=comm)

    barrier_jit = jit(barrier_fn)

    token = gen_token()
    new_token = barrier_jit(token)

    # Token should have the same shape and dtype
    assert new_token.shape == token.shape
    assert new_token.dtype == token.dtype


@pytest.mark.mpi(min_size=2)
def test_barrier_grad(
    array_shape: tuple,
    dtype: DTypeLike,
):
    """Test that barrier passes gradients unchanged (acts as identity)"""

    def func(x):
        token = gen_token()
        # Add x to token so the output depends on x
        token = token + x
        token = barrier(token, comm=comm)
        return jnp.sum(token)

    x = jnp.ones(array_shape, dtype)
    grad_fn = grad(func)
    grad_x = grad_fn(x)

    # Since the function is sum(x + token), grad w.r.t. x is all ones
    assert jnp.allclose(grad_x, jnp.ones_like(x))
