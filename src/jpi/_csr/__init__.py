import jax

# Assume sparsejax is a module with FFI registrations
from jpi import backend

# Register FFI targets
for name, target in backend.registrations().items():
    jax.ffi.register_ffi_target(name, target)
