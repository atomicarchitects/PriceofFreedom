#  python benchmark_tp.py --device=gpu --max_ell=8 --irreps_type=MIMO --tensor_product_type=CGTP-dense

import sys
sys.path.append("../")
from src.tensor_products.functional import clebsch_gordan_tensor_product_dense, clebsch_gordan_tensor_product_sparse, gaunt_tensor_product_s2grid, gaunt_tensor_product_fourier_2D
from benchmarking.utils.fast_flops import flops_counter

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp
from typing import Union, List
import e3nn_jax as e3nn
import flax.linen as nn
import numpy as np
from functools import partial
import time
from absl import flags
import sys

flags.DEFINE_enum("device", "gpu", ["gpu", "cpu"], "Device to run on")
flags.DEFINE_integer("lmax", 2, "lmax")
flags.DEFINE_enum(
    "irreps_type", "MIMO", ["MIMO", "SIMO", "SISO"], "Input/Output Irreps"
)
flags.DEFINE_enum(
    "tensor_product_type",
    "CGTP-dense",
    ["CGTP-dense", "CGTP-sparse", "GTP-grid", "GTP-fourier"],
    "Tensor Product Types",
)
flags.DEFINE_bool(
    "ncu_flag",
    False,
   "Flag for Nsight Compute Benchmarking"
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

WARMUP = 10
TRIALS = 100

# Set device globally
jax.config.update("jax_platform_name", FLAGS.device)
assert jax.devices()[0].platform == FLAGS.device

irreps_mapper = {
    "SISO": lambda lmax: (e3nn.Irreps(f"{lmax}{'e' if lmax % 2 == 0 else 'o'}"),
                         e3nn.Irreps(f"{lmax}{'e' if lmax % 2 == 0 else 'o'}")),
    "SIMO": lambda lmax: (
        e3nn.Irreps(f"{lmax}{'e' if lmax % 2 == 0 else 'o'}"),
        e3nn.s2_irreps(2*lmax)),
    "MIMO": lambda lmax: (
        e3nn.s2_irreps(lmax),
        e3nn.s2_irreps(2*lmax),
    ),
}

def tp_initializer(tp_type, output_irreps):    
    if tp_type == "CGTP-dense":
        return partial(clebsch_gordan_tensor_product_dense, filter_ir_out=output_irreps)
    elif tp_type == "CGTP-sparse":
        return partial(clebsch_gordan_tensor_product_sparse, filter_ir_out=output_irreps)
    elif tp_type == "GTP-grid":
        return partial(gaunt_tensor_product_s2grid, filter_ir_out=output_irreps, res_beta=90, res_alpha=89,  p_val1=1, p_val2=1, s2grid_fft=False, quadrature="gausslegendre")
    elif tp_type == "GTP-fourier":
        return partial(gaunt_tensor_product_fourier_2D, filter_ir_out=output_irreps, res_theta=90, res_phi=180, convolution_type="direct") # Remember to try with fft as well"
    else:
        raise ValueError(f"{tp_type} not supported")

@flops_counter
def func_flops(func, *args):
    return func(*args)

def benchmark_per_lmax(lmax: int, tp_type: str, irreps_type: str):
    input_irreps, output_irreps = irreps_mapper[irreps_type](lmax)
    x = e3nn.normal(input_irreps, jax.random.PRNGKey(0))
    y = e3nn.normal(input_irreps, jax.random.PRNGKey(1))
    TP = tp_initializer(tp_type, output_irreps)
    TP = jax.jit(TP)
    print(f"irreps_type {FLAGS.irreps_type} tensor_product_type {FLAGS.tensor_product_type} lmax {FLAGS.lmax}")
    start = time.process_time()
    result = TP(x, y)
    result.array.block_until_ready()
    if FLAGS.ncu_flag:
        func_flops(TP, x, y)
    else:
        print(f"Compiling took {(time.process_time() - start):.3f} s")
        timings = []
        for _ in range(TRIALS):
            start = time.process_time()
            result = TP(x, y)
            result.array.block_until_ready()
            timings.append(time.process_time() - start)
            
        print(f"Walltime {np.mean(timings):.6f} s")
def main():
    benchmark_per_lmax(
        FLAGS.lmax, FLAGS.tensor_product_type, FLAGS.irreps_type
    )

if __name__ == "__main__":
    main()