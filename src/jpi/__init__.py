import jax

from jpi import backend
from jpi.allgather import allgather
from jpi.allreduce import allreduce
from jpi.bcast import bcast
from jpi.barrier import barrier
from jpi.gather import gather
from jpi.scatter import scatter
from jpi.send import send
from jpi.recv import recv

from jpi.token import gen_token


# WARNING: Unsure if this should be included
jax.config.update("jax_enable_x64", True)

# Register FFI targets
for name, target in backend.registrations().items():
    jax.ffi.register_ffi_target(name, target)

__all__ = [
    "allgather",
    "allreduce",
    "bcast",
    "barrier",
    "gen_token",
    "gather",
    "scatter",
    "send",
    "recv",
]
