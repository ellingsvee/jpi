import sys
from pathlib import Path

# Add tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import jax.numpy as jnp
from jax._src.typing import DTypeLike
from mpi4py import MPI

DTYPE = [
    pytest.param(jnp.float64, id="float64"),
    pytest.param(jnp.float32, id="float32"),
]

ARRAY_SHAPE = [
    pytest.param((5,), id="array_shape=(5,)"),
    pytest.param((2, 2), id="array_shape=(2,2)"),
    pytest.param((2, 4, 2), id="array_shape=(2,4,2)"),
]

OP = [
    pytest.param(MPI.SUM, id="MPI.SUM"),
    pytest.param(MPI.PROD, id="MPI.PROD"),
    pytest.param(MPI.MAX, id="MPI.MAX"),
    pytest.param(MPI.MIN, id="MPI.MIN"),
]


@pytest.fixture(params=ARRAY_SHAPE, autouse=True)
def array_shape(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=DTYPE, autouse=True)
def dtype(request: pytest.FixtureRequest) -> DTypeLike:
    return request.param


@pytest.fixture(params=OP, autouse=True)
def op(request: pytest.FixtureRequest):
    return request.param
