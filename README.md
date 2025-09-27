# Sparsejax

An implementation of a sparse CSR matrix in JAX. This project was intended to see how to handle the limitations of sparse matrices in JAX. Notably, to get jittable operations, we often need to know the number of non-zero elements. This is handled by running a non-jitted callback that counts the numbers the first time we run.
