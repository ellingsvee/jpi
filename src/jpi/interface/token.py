# Internal token for ordering MPI operations
import jax.numpy as jnp


def gen_token():
    return jnp.zeros((), dtype=jnp.float32)
