import nvtx
from functools import wraps

def is_jax_tensor(tensor):
    """
    Check if the given tensor is a JAX tensor.
    """
    return (str(type(tensor)) == "<class 'jaxlib.xla_extension.DeviceArray'>") | (str(type(tensor)) == "<class 'jaxlib.xla_extension.ArrayImpl'>")

def is_jax_dynamic_tensor(tensor):
    """
    Check if the given tensor is a Dynamic JAX tensor so that block_until_ready can be ignored.
    """
    return str(type(tensor)) == "<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>"

def is_torch_tensor(tensor):
    """
    Check if the given tensor is a PyTorch tensor.
    """
    return str(type(tensor)).startswith("<class 'torch.Tensor")

def is_e3nn_tensor(tensor):
    """
    Check if given tensor corresponds to e3nn IrrepsArray
    """
    return str(type(tensor)) == "<class 'e3nn_jax._src.irreps_array.IrrepsArray'>"

def flops_counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        # A few warmup laps to cleanse the JIT
        for _ in range(10):
            result = func(*args, **kwargs)

            if is_jax_tensor(result):
                import jax
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
            elif is_e3nn_tensor(result):
                import jax
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result.array)
            elif is_torch_tensor(result):
                import torch
                torch.cuda.synchronize()
            elif is_jax_dynamic_tensor(result):
                # Doesnt make sense to block traced tensors
                pass
            else:
                raise ValueError(f"{type(result)} not supported")

        nvtx_range = nvtx.start_range(f"profile")
        result = func(*args, **kwargs)
        nvtx.end_range(nvtx_range)
        return result
    return wrapper