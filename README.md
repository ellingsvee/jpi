# JPI

JPI (JAX Parallel Interface) is a library for distributed computing with [JAX](https://github.com/google/jax) using MPI. It provides composable primitives for parallel operations that integrate seamlessly with JAX's transformation system, enabling efficient distributed scientific computing.

## Features

- **Complete MPI collective operations**: Implementations of `allreduce`, `allgather`, `scatter`, `gather`, `broadcast`, and `barrier`. More can easily be added.
- **JAX transformation compatibility**: Full support for `jit`, `grad`, `vmap`, and other JAX transformations
- **Automatic differentiation**: Custom VJP definitions ensure correct gradient computation through parallel operations
- **Interoperability with mpi4py**: Uses `mpi4py` communicators, allowing easy integration with existing MPI-based codebases.
- **Token-based synchronization**: As JAX and XLA operate on the assumption that all primitives are pure functions without side effects, the compiler is in principle free to re-order operations. Inspired by [mpi4jax](https://github.com/mpi4jax/mpi4jax/tree/main), this is handeled by introducing a fake data dependency between subsequent calls using tokens.
- **Implemented with JAX FFI**: The MPI operations are implemented in C++ and interfaced with JAX using the Foreign Function Interface (FFI) for performance. There is no copying of data between JAX and the MPI backend, ensuring low overhead.

## Current Limitations

- **CPU-only support**: GPU operations are not yet implemented. However, this can be added in future versions by implementing FFI calls on GPU as described in the [FFI calls documentation](https://docs.jax.dev/en/latest/ffi.html#ffi-calls-on-a-gpu).
- **Limited MPI operations**: Only a subset of MPI collective operations are implemented.
- **Development stage**: API may change in future versions

## Installation
### Prerequisites
Installing currently requires some system-level dependencies. Make sure these are installed:
- `uv`: Recommended package manager. See [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).
- `Python3 >=3.13`: See [Installing python](https://docs.astral.sh/uv/guides/install-python/) for installing Python using `uv`. 
- `git`: See [git downloads](https://git-scm.com/downloads).
- `OpenMPI`: See [OpenMPI documentation](https://docs.open-mpi.org/en/v5.0.x/index.html). You might need to update the CMakeLists.txt file to point to the correct MPI installation.

### Building the project
As the MPI operations are implemented in C++, the project needs to be built before use. This is handled automatically when installing with `uv`. Install the project with:
```bash
# Clone the repository
git clone https://github.com/ellingsvee/jpi.git
cd jpi

# Install with uv
uv sync 
uv build
```

### Modifying the C++ backend
If you make changes to the C++ code, you need to rebuild the project. This can be done with:
```bash
uv sync --reinstall
uv build
```




## Usage

```python
import jax
import jax.numpy as jnp
from mpi4py import MPI
from jpi import allreduce, gen_token

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Each rank starts with its own data
data = jnp.array([comm.rank], dtype=jnp.float32)


# Some function that uses allreduce
def func(data):
    token = gen_token()
    result, _ = allreduce(data, token, op=MPI.SUM, comm=comm)

    # Each rank can do something different
    if rank == 0:
        result = result * 2
    return jnp.sum(result)


# JIT and grad the function
func_jit = jax.jit(func)
func_grad = jax.grad(func_jit)

# Compute result and gradient
result = func_jit(data)
grad_result = func_grad(data)
print(f"Rank {comm.rank} has result {result} and gradient {grad_result}")
```
Run the above code with MPI using:
```bash
mpirun -np 4 uv run python examples/intro_example.py
```

## Testing
Tests are implemented using `pytest`. To run the tests with MPI use:
```bash
mpirun -np 4 uv run pytest --with-mpi 
```

## License
MIT License. See `LICENSE` file for details.

## Alternatives
This project is inspired by the great [mpi4jax](https://github.com/mpi4jax/mpi4jax).  Built using `mpi4py.libmpi` to exposes MPI C primitives as Cython callables, mpi4jax is currently more mature and has more features. JPI aims to provide a simpler and more extensible framework for integrating MPI with JAX.  Additionally, building on top of JAX's FFI allows XLA to better optimize the C++ backend for performance.
 