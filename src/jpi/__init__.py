import jax
from jpi import backend


jax.config.update("jax_enable_x64", True)

# Register FFI targets
for name, target in backend.registrations().items():
    jax.ffi.register_ffi_target(name, target)
