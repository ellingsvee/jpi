import jax
from jpi import backend
from jpi.interface import bcast


jax.config.update("jax_enable_x64", True)

# Register FFI targets
for name, target in backend.registrations().items():
    jax.ffi.register_ffi_target(name, target)

__all__ = ["bcast"]
