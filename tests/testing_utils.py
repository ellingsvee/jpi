import jax
from jax._src.typing import DTypeLike


def generate_array(
    array_shape: tuple,
    dtype: DTypeLike,
) -> jax.Array:
    key = jax.random.PRNGKey(0)
    return jax.random.uniform(key, array_shape, dtype=dtype)
