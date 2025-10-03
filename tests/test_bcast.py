import pytest

import jax.numpy as jnp
from jax._src.typing import DTypeLike

from jpi.interface.token import gen_token
from jpi.interface.bcast import bcast

from jpi.testing_utils import generate_array


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(min_size=2)
def test_allgather(
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


# if __name__ == "__main__":
#     test_allgather((5,), dtype=jnp.float64)
#     print("success!")
