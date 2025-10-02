# from functools import partial
# import jax
# from jpi.interface.bcast import _bcast_impl
# from jpi.mpi import rank, size
#
#
# def _reduce_impl(x: jax.Array, root: int, op: int):
#     y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
#     input_output_aliases = {0: 0}  # alias input and output buffers
#     out = jax.ffi.ffi_call(
#         "reduce",
#         (y_type,),
#         vmap_method="sequential",
#         input_output_aliases=input_output_aliases,
#     )(x, root=root, rank=rank, size=size, op=op)[0]  # Unpacking the tuple at the end
#     return out
#
#
# @partial(jax.custom_vjp, nondiff_argnums=(1, 2))
# def reduce(x: jax.Array, root: int, op: int):
#     return _reduce_impl(x, root, op)
#
#
# def reduce_fwd(x: jax.Array, root: int, op: int):
#     y = _reduce_impl(x, root, op)
#     return y, (x, op)
#
#
# def reduce_bwd(root: int, op: int, res: tuple, g: jax.Array):
#     x, op = res
#     if op == 0:  # MPI_SUM
#         # For sum, gradient is broadcast from root to all ranks
#         return (_bcast_impl(g, root),)
#     # elif op == 1:  # MPI_PROD
#     #     # For product, gradient is y / x_i (on each rank)
#     #     y = _reduce_impl(x, root, op)
#     #     # Broadcast y to all ranks to compute gradient
#     #     y = _bcast_impl(y, root)
#     #     grad = jnp.where(x != 0, y / x, 0)  # Avoid division by zero
#     #     return (grad,)
#     # elif op in (2, 3):  # MPI_MIN, MPI_MAX
#     #     # For min/max, gradient is 1 where x is min/max, 0 elsewhere
#     #     y = _reduce_impl(x, root, op)
#     #     # Broadcast y to all ranks
#     #     y = _bcast_impl(y, root)
#     #     grad = jnp.where(x == y, 1.0, 0.0)
#     #     return (grad,)
#     else:
#         raise ValueError(f"Unsupported op: {op}")
#
#
# reduce.defvjp(reduce_fwd, reduce_bwd)
