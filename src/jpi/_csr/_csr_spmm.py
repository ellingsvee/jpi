# import time
#
# from functools import partial
# import jax
# import jax.numpy as jnp
#
# # Assume sparsejax is a module with FFI registrations
# from sparsejax._csr._csr_to_other import csr_to_csc
# from sparsejax._csr._csr_utils import csr_sorting
#
#
# def _csr_spmm_impl(Ap, Aj, Ax, Bp, Bj, Bx, n_rows, n_cols, nnz):
#     Cp_shape = (n_rows + 1,)
#     Cj_shape = (nnz,)
#     Cx_shape = (nnz,)
#     Cp_type = jax.ShapeDtypeStruct(Cp_shape, jnp.int64)
#     Cj_type = jax.ShapeDtypeStruct(Cj_shape, jnp.int64)
#     Cx_type = jax.ShapeDtypeStruct(Cx_shape, jnp.float32)
#     nnz_type = jax.ShapeDtypeStruct((1,), jnp.int64)
#     return jax.ffi.ffi_call(
#         "csr_spmm",
#         (Cp_type, Cj_type, Cx_type, nnz_type),
#         vmap_method="sequential",
#     )(Ap, Aj, Ax, Bp, Bj, Bx, n_cols=n_cols)
#
#
# @partial(jax.custom_vjp)
# @partial(jax.jit, static_argnums=(6, 7, 8))
# def csr_spmm(Ap, Aj, Ax, Bp, Bj, Bx, n_rows, n_cols, nnz):
#     start_time = time.time()
#     Cp, Cj, Cx, nnz_out = _csr_spmm_impl(Ap, Aj, Ax, Bp, Bj, Bx, n_rows, n_cols, nnz)
#     end_time = time.time()
#     print(f"FW-pass: Matmul took {end_time - start_time} sec")
#
#     # TODO: Figure out if we can avoid this
#     # Applying the sorting
#
#     start_time = time.time()
#     Cp, Cj, Cx = csr_sorting(Cp, Cj, Cx, n_rows)
#     end_time = time.time()
#     print(f"FW-pass: Sorting took {end_time - start_time} sec")
#
#     return Cp, Cj, Cx, nnz_out
#
#
# def csr_spmm_fwd(Ap, Aj, Ax, Bp, Bj, Bx, n_rows, n_cols, nnz):
#     Cp, Cj, Cx, nnz_out = _csr_spmm_impl(Ap, Aj, Ax, Bp, Bj, Bx, n_rows, n_cols, nnz)
#     Cp, Cj, Cx = csr_sorting(Cp, Cj, Cx, n_rows)
#     res = (Cp, Cj, Ap, Aj, Ax, Bp, Bj, Bx, n_rows, n_cols, nnz)
#     return (Cp, Cj, Cx, nnz_out), res
#
#
# def csr_spmm_bwd(res, ct):
#     Cp, Cj, Ap, Aj, Ax, Bp, Bj, Bx, n_rows, n_cols, nnz_C = res
#     nnz_A = Aj.shape[0]
#     nnz_B = Bj.shape[0]
#
#     # Getting the adjoint of C
#     _, _, GCx, _ = ct
#
#     # The nnz type
#     nnz_type = jax.ShapeDtypeStruct((1,), jnp.int64)
#
#     # Computing the adjoint of A
#     GAp_type = jax.ShapeDtypeStruct((n_rows + 1,), jnp.int64)
#     GAj_type = jax.ShapeDtypeStruct((nnz_A,), jnp.int64)
#     GAx_type = jax.ShapeDtypeStruct((nnz_A,), jnp.float32)
#     Btp, Bti, Btx = csr_to_csc(
#         Bp, Bj, Bx, n_cols, n_rows
#     )  # Use the CSC representation for B.T
#
#     # start_time = time.time()
#     # GAp_type = jax.ShapeDtypeStruct((n_rows + 1,), jnp.int64)
#     # GAj_type = jax.ShapeDtypeStruct((nnz_C * nnz_B // 10,), jnp.int64)
#     # GAx_type = jax.ShapeDtypeStruct((nnz_C * nnz_B // 10,), jnp.float32)
#     # Btp, Bti, Btx = csr_to_csc(
#     #     Bp, Bj, Bx, n_cols, n_rows
#     # )  # Use the CSC representation for B.T
#     # GAp, GAj, GAx, _ = jax.ffi.ffi_call(
#     #     "csr_spmm",  # Use the masked version for gradients.
#     #     (GAp_type, GAj_type, GAx_type, nnz_type),
#     #     vmap_method="sequential",
#     # )(
#     #     Cp,
#     #     Cj,
#     #     GCx,
#     #     Btp,
#     #     Bti,
#     #     Btx,
#     #     # Ap,  # Use A as mask
#     #     # Aj,  # Use A as mask
#     #     n_cols=n_cols,
#     # )
#     # end_time = time.time()
#     # print(f"BW-pass: Unmasked matmul took {end_time - start_time} sec")
#
#     # Computing the adjoint of A
#     GAp_type = jax.ShapeDtypeStruct((n_rows + 1,), jnp.int64)
#     GAj_type = jax.ShapeDtypeStruct((nnz_A,), jnp.int64)
#     GAx_type = jax.ShapeDtypeStruct((nnz_A,), jnp.float32)
#     Btp, Bti, Btx = csr_to_csc(
#         Bp, Bj, Bx, n_cols, n_rows
#     )  # Use the CSC representation for B.T
#
#     start_time = time.time()
#     GAp, GAj, GAx, _ = jax.ffi.ffi_call(
#         "csr_spmm_masked",  # Use the masked version for gradients.
#         (GAp_type, GAj_type, GAx_type, nnz_type),
#         vmap_method="sequential",
#     )(
#         Cp,
#         Cj,
#         GCx,
#         Btp,
#         Bti,
#         Btx,
#         Ap,  # Use A as mask
#         Aj,  # Use A as mask
#         n_cols=n_cols,
#     )
#     end_time = time.time()
#     print(f"BW-pass: Matmul took {end_time - start_time} sec")
#
#     # TODO: Would be nice to avoid the sorting
#     start_time = time.time()
#     GAp, GAj, GAx = csr_sorting(GAp, GAj, GAx, n_rows)
#     end_time = time.time()
#     print(f"BW-pass: Sorting took {end_time - start_time} sec")
#
#     # Computing the adjoint of B
#     GBp_type = jax.ShapeDtypeStruct((n_cols + 1,), jnp.int64)
#     GBj_type = jax.ShapeDtypeStruct((nnz_B,), jnp.int64)
#     GBx_type = jax.ShapeDtypeStruct((nnz_B,), jnp.float32)
#     Atp, Ati, _ = csr_to_csc(
#         Ap, Aj, Ax, n_rows, n_cols
#     )  # Use the CSC representation for A.T
#
#     GBp, GBj, GBx, _ = jax.ffi.ffi_call(
#         "csr_spmm_masked",  # Use the masked version for gradients
#         (GBp_type, GBj_type, GBx_type, nnz_type),
#         vmap_method="sequential",
#     )(
#         Atp,
#         Ati,  # swapped Ap/Aj to represent A^T per your kernel
#         Ax,
#         Cp,
#         Cj,
#         GCx,
#         Bp,  # Use B as mask
#         Bj,  # Use B as mask
#         n_cols=n_rows,  # adjust dims if necessary when transposing
#     )
#
#     # TODO: Would be nice to avoid the sorting
#     GBp, GBj, GBx = csr_sorting(GBp, GBj, GBx, n_cols)
#
#     return (None, None, GAx, None, None, GBx, None, None, None)
#
#
# csr_spmm.defvjp(csr_spmm_fwd, csr_spmm_bwd)
