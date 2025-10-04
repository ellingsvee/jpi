import jax
from jpi import backend
from jpi import interface
from jpi.interface import *


# WARNING: Unsure if this should be included
jax.config.update("jax_enable_x64", True)

# Register FFI targets
for name, target in backend.registrations().items():
    jax.ffi.register_ffi_target(name, target)
