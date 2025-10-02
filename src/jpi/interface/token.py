# Internal token for ordering MPI operations
import jax.numpy as jnp


class _TokenManager:
    """Manages a global token for ordering MPI operations."""

    def __init__(self):
        self._token = None

    def get_token(self):
        """Get the current token, creating one if needed."""
        if self._token is None:
            # Create a zero-dimensional array as the initial token
            self._token = jnp.zeros((), dtype=jnp.float32)
        return self._token

    def update_token(self, new_token):
        """Update the global token."""
        self._token = new_token


# Global token manager instance
_token_manager = _TokenManager()
