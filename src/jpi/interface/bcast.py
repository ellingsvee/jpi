from functools import partial
import jax


def _bcast_impl(x, root):
    y_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    input_output_aliases = {0: 0}  # alias input and output buffers
    return jax.ffi.ffi_call(
        "bcast",
        (y_type,),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(x, root=root)[0]  # Unpacking the tuple at the end


@partial(jax.custom_vjp, nondiff_argnums=(1,))
@partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
def bcast(x, root):
    return _bcast_impl(x, root)


def bcast_fwd(x, root):
    y = _bcast_impl(x, root)
    return y, None  # No aux data needed for backward pass


def bcast_bwd(root, _, g):
    return (_bcast_impl(g, root),)  # Return as a tuple


bcast.defvjp(bcast_fwd, bcast_bwd)
