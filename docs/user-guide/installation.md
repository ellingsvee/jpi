# Installation

## Requirements

JPI requires:

- Python ≥ 3.13
- JAX ≥ 0.7.2
- MPI4Py ≥ 4.1.0
- An MPI implementation (OpenMPI, MPICH, etc.)

## Installing MPI

### macOS (using Homebrew)

```bash
brew install open-mpi
```

### Ubuntu/Debian

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

### Other Systems

Refer to your system's package manager or install from source:
- [OpenMPI](https://www.open-mpi.org/)
- [MPICH](https://www.mpich.org/)

## Installing JPI

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/ellingsvee/jpi.git
   cd jpi
   ```

2. Install using uv (recommended):
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

## Verifying Installation

Test your installation by running:

```bash
mpirun -n 2 python -c "
import jax
import jpi
from jpi.interface import barrier
from jpi.interface.token import make_token

print(f'Rank {jax.process_index()} ready')
token = make_token()
result, token = barrier(token)
print(f'Rank {jax.process_index()} synchronized')
"
```

If everything is working correctly, you should see output from both MPI processes indicating successful synchronization.