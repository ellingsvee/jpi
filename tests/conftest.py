import pytest
import jax.numpy as jnp
from jax._src.typing import DTypeLike

DTYPE = [
    pytest.param(jnp.float64, id="float64"),
    pytest.param(jnp.float32, id="float32"),
]

ARRAY_SHAPE = [
    pytest.param((5,), id="array_shape=(5,)"),
]


@pytest.fixture(params=ARRAY_SHAPE, autouse=True)
def array_shape(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=DTYPE, autouse=True)
def dtype(request: pytest.FixtureRequest) -> DTypeLike:
    return request.param
