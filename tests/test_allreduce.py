import pytest
from mpi4py import MPI

from jax import grad, jit
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi.interface.token import gen_token
from jpi.interface.allreduce import allreduce

from jpi.testing_utils import generate_array


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_allreduce(
    array_shape: tuple,
    dtype: DTypeLike,
    op,
):
    """Test allreduce with various operations"""
    # Generate appropriate data for each operation
    if op == MPI.PROD:
        arr = jnp.abs(generate_array(array_shape, dtype)) + 1.0
        expected = arr**size
    elif op == MPI.SUM:
        arr = generate_array(array_shape, dtype)
        expected = arr * size
    elif op == MPI.MAX:
        arr = generate_array(array_shape, dtype) + rank
        expected = generate_array(array_shape, dtype) + (size - 1)
    elif op == MPI.MIN:
        arr = generate_array(array_shape, dtype) + rank
        expected = generate_array(array_shape, dtype) + 0
    else:
        raise ValueError(f"Unsupported operation: {op}")

    token = gen_token()
    y, _ = allreduce(arr, token, op, comm=comm)

    assert jnp.allclose(y, expected)


@pytest.mark.mpi(min_size=2)
def test_allreduce_jit(
    array_shape: tuple,
    dtype: DTypeLike,
    op,
):
    """Test allreduce with JIT compilation"""
    # Generate appropriate data for each operation
    if op == MPI.PROD:
        arr = jnp.abs(generate_array(array_shape, dtype)) + 1.0
        expected = arr**size
    elif op == MPI.SUM:
        arr = generate_array(array_shape, dtype)
        expected = arr * size
    elif op == MPI.MAX:
        arr = generate_array(array_shape, dtype) + rank
        expected = generate_array(array_shape, dtype) + (size - 1)
    elif op == MPI.MIN:
        arr = generate_array(array_shape, dtype) + rank
        expected = generate_array(array_shape, dtype) + 0
    else:
        raise ValueError(f"Unsupported operation: {op}")

    def allreduce_fn(x):
        token = gen_token()
        y, _ = allreduce(x, token, op, comm=comm)
        return y

    allreduce_jit = jit(allreduce_fn)
    y = allreduce_jit(arr)

    assert jnp.allclose(y, expected)


@pytest.mark.mpi(min_size=2)
def test_allreduce_grad(
    array_shape: tuple,
    dtype: DTypeLike,
    op,
):
    """Test that allreduce gradient works correctly for implemented operations"""
    # Generate appropriate data for each operation
    if op == MPI.PROD:
        arr = jnp.abs(generate_array(array_shape, dtype)) + 1.0
        expected_grad_fn = lambda arr, size, rank: arr ** (size - 1)
    elif op == MPI.SUM:
        arr = generate_array(array_shape, dtype)
        expected_grad_fn = lambda arr, size, rank: jnp.ones_like(arr) * size
    elif op == MPI.MAX:
        arr = generate_array(array_shape, dtype) + rank
        expected_grad_fn = (
            lambda arr, size, rank: jnp.ones_like(arr)
            if rank == size - 1
            else jnp.zeros_like(arr)
        )
    elif op == MPI.MIN:
        arr = generate_array(array_shape, dtype) + rank
        expected_grad_fn = (
            lambda arr, size, rank: jnp.ones_like(arr)
            if rank == 0
            else jnp.zeros_like(arr)
        )
    else:
        pytest.skip(f"Gradient test not implemented for operation: {op}")

    def func(x):
        token = gen_token()
        y, _ = allreduce(x, token, op, comm=comm)
        return jnp.sum(y)

    grad_fn_jax = grad(func)
    grad_x = grad_fn_jax(arr)

    expected = expected_grad_fn(arr, size, rank)
    assert jnp.allclose(grad_x, expected)
