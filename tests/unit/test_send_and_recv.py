import pytest
from mpi4py import MPI

from jax import grad, jit
import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi import gen_token
from jpi import send, recv

from tests.testing_utils import generate_array


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_send_and_recv(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Each rank creates different data
    arr = generate_array(array_shape, dtype) + rank

    token = gen_token()

    if rank == 0:
        dest = 1
    else:
        dest = 0

    arr, token = send(arr, token, dest=dest, tag=0, comm=comm)

    y = jnp.empty_like(arr)
    y, token = recv(y, token, source=dest, tag=0, comm=comm)

    if rank == 0 or rank == 1:
        # Each rank should receive data from the other rank
        expected = generate_array(array_shape, dtype) + (dest)
        assert jnp.allclose(y, expected)


@pytest.mark.mpi(min_size=2)
def test_send_and_recv_jit(
    array_shape: tuple,
    dtype: DTypeLike,
):
    # Each rank creates different data
    arr = generate_array(array_shape, dtype) + rank

    def send_recv_fn(x):
        token = gen_token()

        if rank == 0:
            dest = 1
        else:
            dest = 0

        # Send and receive
        x, token = send(x, token, dest=dest, tag=0, comm=comm)
        y = jnp.empty_like(x)
        y, token = recv(y, token, source=dest, tag=0, comm=comm)
        return y

    send_recv_jit = jit(send_recv_fn)
    y = send_recv_jit(arr)

    if rank == 0 or rank == 1:
        # Each rank should receive data from the other rank
        dest = 1 if rank == 0 else 0
        expected = generate_array(array_shape, dtype) + dest
        assert jnp.allclose(y, expected)


@pytest.mark.mpi(min_size=2)
def test_send_and_recv_grad(
    array_shape: tuple,
    dtype: DTypeLike,
):
    arr = generate_array(array_shape, dtype) + rank

    def func(x):
        token = gen_token()

        if rank == 0:
            dest = 1
        else:
            dest = 0

        x = x * rank  # Make function rank-dependent

        # Send and receive
        x, token = send(x, token, dest=dest, tag=0, comm=comm)
        y = jnp.empty_like(x)
        y, token = recv(y, token, source=dest, tag=0, comm=comm)

        # Return scalar for gradient computation
        return jnp.sum(y)

    grad_fn = grad(func)
    grad_x = grad_fn(arr)

    if rank == 0 or rank == 1:
        # The gradient of (x * rank) w.r.t. x is rank
        # Gradient flows: sum(y) → recv → send → (x * rank) → x
        # So ∂loss/∂x = ∂loss/∂y * ∂(x*rank)/∂x = 1 * rank = rank
        expected = jnp.ones_like(arr) * rank
        assert jnp.allclose(grad_x, expected)


if __name__ == "__main__":
    test_send_and_recv((2, 2), jnp.float32)
