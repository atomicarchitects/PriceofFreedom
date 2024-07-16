import nvtx
from functools import wraps

def flops_counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from ctypes import cdll
        libcudart = cdll.LoadLibrary("libcudart.so")
        libcudart.cudaProfilerStart()
        nvtx_range = nvtx.start_range("profile")
        result = func(*args, **kwargs)
        nvtx.end_range(nvtx_range)
        libcudart.cudaProfilerStop()
        return result
    return wrapper