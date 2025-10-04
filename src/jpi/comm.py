def get_default_comm():
    """Get the default MPI communicator (MPI.COMM_WORLD)."""
    from mpi4py import MPI

    return MPI.COMM_WORLD.Clone()
