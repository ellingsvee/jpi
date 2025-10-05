from typing import TypeAlias
from mpi4py import MPI

Comm: TypeAlias = MPI.Comm


def get_default_comm() -> Comm:
    """Get the default MPI communicator (MPI.COMM_WORLD)."""
    from mpi4py import MPI

    return MPI.COMM_WORLD.Clone()
