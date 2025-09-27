import jax
jax.config.update("jax_enable_x64", True)  # Need this for the FFI

# Assume sparsejax is a module with FFI registrations
from jpi import backend

# Register FFI targets
for name, target in backend.registrations().items():
    jax.ffi.register_ffi_target(name, target)


def main() -> None:
    print("JPI module loaded")

if __name__ == "__main__":
    main()