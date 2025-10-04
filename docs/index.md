# JPI - JAX Parallel Interface

JPI provides MPI-based parallel operations for JAX, enabling distributed computing with JAX arrays across multiple processes.

## Overview

JPI offers a collection of collective communication operations optimized for JAX:

- **allgather**: Gather arrays from all processes 
- **allreduce**: Reduce arrays across all processes
- **bcast**: Broadcast arrays from one process to all others
- **barrier**: Synchronize all processes

All operations are designed to work seamlessly with JAX's compilation and differentiation systems.

## Key Features

- **JAX Integration**: Full compatibility with `jax.jit`, `jax.grad`, and other JAX transformations
- **MPI Backend**: Built on MPI for high-performance distributed computing
- **Custom VJP Support**: Differentiable collective operations where applicable
- **Type Safety**: Proper handling of JAX array shapes and dtypes

## Quick Example

```python
import jax
import jax.numpy as jnp
from jpi.interface import allreduce, bcast
from jpi.interface.token import make_token

# Create some data
x = jnp.array([1.0, 2.0, 3.0])
token = make_token()

# Broadcast from rank 0
result, token = bcast(x, token, root=0)

# Sum across all processes
sum_result, token = allreduce(x, token, op="sum")
```

## Installation

See the [Installation Guide](user-guide/installation.md) for detailed setup instructions.

## Getting Started

Check out the [Quick Start Guide](user-guide/quickstart.md) to begin using JPI in your projects.