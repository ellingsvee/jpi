# JPI

JPI (JAX Parallel Interface) is a library for distributed computing with [JAX](https://github.com/google/jax) using MPI. It provides simple, composable primitives for parallel operations, enabling scalable scientific computing and machine learning workflows.

## Features

- **MPI-based parallelism**: Use multiple processes for distributed computation.
- **JAX integration**: All primitives are compatible with JAX transformations (`jit`, `grad`, etc.).
- **Custom gradients**: Support for autodiff through custom VJP definitions.
- **Simple API**: Intuitive functions for common parallel operations.

## TODO
- **GPU**: Currently only CPUs are supported. However, it should be easy to extend to through FFI (see [FFI documentation](https://docs.jax.dev/en/latest/ffi.html#ffi-calls-on-a-gpu)).
