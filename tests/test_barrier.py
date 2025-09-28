import time
import sys
import jax
import jax.numpy as jnp
from jpi.mpi import rank, comm


def input_array():
    if rank == 0:
        return jnp.arange(5, dtype=jnp.float32)
    else:
        return jnp.zeros(5, dtype=jnp.float32)


@jax.jit
def function_to_be_jitted(x):
    # Rank 0 performs a quick task
    if rank == 0:
        print(f"Rank {rank}: Starting quick task...")
        time.sleep(0.5)
        print(f"Rank {rank}: Finished quick task.")

    # Rank 1 performs a long task
    elif rank == 1:
        print(f"Rank {rank}: Starting LONG task...")
        time.sleep(2.0)
        print(f"Rank {rank}: Finished LONG task.")

    # Ensure all output is flushed before the barrier (good practice)
    sys.stdout.flush()

    # --- The Barrier Call ---
    print(f"Rank {rank}: Reached the barrier. Waiting...")
    comm.barrier()
    # This code block will only start after the longest process (Rank 1) is finished.
    print(f"Rank {rank}: Passed the barrier. All processes are now synchronized.")

    return jnp.sum(x)
    # --- End of Barrier ---


if __name__ == "__main__":
    print("Starting the JAX MPI example...")
    comm.barrier()  # Initial barrier to sync before starting
    print(f"Rank {rank}: All processes synchronized at start.")

    x = input_array()
    print(f"Rank {rank}: Input array: x = {x}")
    y = function_to_be_jitted(x)
    print(f"Rank {rank}: Result after barrier and sum: y = {y}")
