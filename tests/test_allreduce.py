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


# Parametrize operations
@pytest.mark.parametrize(
    "op,data_fn,expected_fn",
    [
        (
            MPI.SUM,
            lambda shape, dtype: generate_array(shape, dtype),
            lambda arr, size, rank: arr * size,
        ),
        (
            MPI.PROD,
            lambda shape, dtype: jnp.abs(generate_array(shape, dtype)) + 1.0,
            lambda arr, size, rank: arr**size,
        ),
        (
            MPI.MAX,
            lambda shape, dtype: generate_array(shape, dtype) + rank,
            lambda arr, size, rank: generate_array(arr.shape, arr.dtype) + (size - 1),
        ),
        (
            MPI.MIN,
            lambda shape, dtype: generate_array(shape, dtype) + rank,
            lambda arr, size, rank: generate_array(arr.shape, arr.dtype) + 0,
        ),
    ],
)
@pytest.mark.mpi(min_size=2)
def test_allreduce(
    array_shape: tuple,
    dtype: DTypeLike,
    op,
    data_fn,
    expected_fn,
):
    """Test allreduce with various operations"""
    arr = data_fn(array_shape, dtype)

    token = gen_token()
    y, _ = allreduce(arr, token, op, comm=comm)

    expected = expected_fn(arr, size, rank)
    rtol = 1e-5 if op == MPI.PROD else 1e-7
    assert jnp.allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize("op", [MPI.SUM, MPI.PROD, MPI.MAX, MPI.MIN])
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
        rtol = 1e-5
    elif op == MPI.SUM:
        arr = generate_array(array_shape, dtype)
        expected = arr * size
        rtol = 1e-7
    elif op == MPI.MAX:
        arr = generate_array(array_shape, dtype) + rank
        expected = generate_array(array_shape, dtype) + (size - 1)
        rtol = 1e-7
    elif op == MPI.MIN:
        arr = generate_array(array_shape, dtype) + rank
        expected = generate_array(array_shape, dtype) + 0
        rtol = 1e-7
    else:
        raise ValueError(f"Unsupported operation: {op}")

    def allreduce_fn(x):
        token = gen_token()
        y, _ = allreduce(x, token, op, comm=comm)
        return y

    allreduce_jit = jit(allreduce_fn)
    y = allreduce_jit(arr)

    assert jnp.allclose(y, expected, rtol=rtol)


@pytest.mark.parametrize(
    "op,data_fn,grad_fn",
    [
        (
            MPI.SUM,
            lambda shape, dtype: generate_array(shape, dtype),
            lambda arr, size, rank: jnp.ones_like(arr) * size,
        ),
        (
            MPI.PROD,
            lambda shape, dtype: jnp.abs(generate_array(shape, dtype)) + 1.0,
            lambda arr, size, rank: arr ** (size - 1),
        ),
        (
            MPI.MAX,
            lambda shape, dtype: generate_array(shape, dtype) + rank,
            lambda arr, size, rank: jnp.ones_like(arr)
            if rank == size - 1
            else jnp.zeros_like(arr),
        ),
        (
            MPI.MIN,
            lambda shape, dtype: generate_array(shape, dtype) + rank,
            lambda arr, size, rank: jnp.ones_like(arr)
            if rank == 0
            else jnp.zeros_like(arr),
        ),
    ],
)
@pytest.mark.mpi(min_size=2)
def test_allreduce_grad(
    array_shape: tuple,
    dtype: DTypeLike,
    op,
    data_fn,
    grad_fn,
):
    """Test that allreduce gradient works correctly for all implemented operations"""
    arr = data_fn(array_shape, dtype)

    def func(x):
        token = gen_token()
        y, _ = allreduce(x, token, op, comm=comm)
        return jnp.sum(y)

    grad_fn_jax = grad(func)
    grad_x = grad_fn_jax(arr)

    expected = grad_fn(arr, size, rank)
    assert jnp.allclose(grad_x, expected)
