import csv
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from freedom.tensor_products.functional import clebsch_gordan_tensor_product_dense, clebsch_gordan_tensor_product_sparse, gaunt_tensor_product_s2grid, gaunt_tensor_product_2D_fourier, vector_gaunt_tensor_product_s2grid, matrix_tensor_product
from freedom.tensor_products.vector_spherical_harmonics import VSHCoeffs

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

flags.DEFINE_enum("device", "gpu", ["gpu", "cpu"], "Device to run on")
flags.DEFINE_integer("lmax", 2, "lmax")
flags.DEFINE_integer("batch", 1, "batch")
flags.DEFINE_enum(
    "irreps_type", "MIMO", ["MIMO", "SIMO", "SISO"], "Input/Output Irreps"
)
flags.DEFINE_enum(
    "tensor_product_type",
    "CGTP-dense",
    ["CGTP-dense", "CGTP-sparse", "GTP-grid", "GTP-fourier", "Matrix-TP"],
    "Tensor Product Types",
)
flags.DEFINE_bool(
    "ncu_flag",
    False,
   "Flag for Nsight Compute Benchmarking"
)
flags.DEFINE_bool(
    "dot_graph",
    False,
   "Flag for dumping DOT graph"
)
flags.DEFINE_bool(
    "lmax_based_grid",
    False,
   "Flag for lmax_based_grid"
)
flags.DEFINE_string(
    "walltime_file",
    "benchmarking/csv/walltime.csv",
    "Path to Walltime file"
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

def tp_initializer(lmax, tp_type, irreps_type):
    if tp_type in ["CGTP-dense", "CGTP-sparse", "GTP-grid", "GTP-fourier", "Matrix-TP"]:
        input_irreps, output_irreps = irreps_mapper[irreps_type](lmax)
        x = e3nn.normal(input_irreps, jax.random.PRNGKey(0), (FLAGS.batch,))
        y = e3nn.normal(input_irreps, jax.random.PRNGKey(1), (FLAGS.batch,))
        if tp_type == "CGTP-dense":
            return x, y, partial(clebsch_gordan_tensor_product_dense, filter_ir_out=output_irreps, irrep_normalization="norm")
        elif tp_type == "CGTP-sparse":
            return x, y, partial(clebsch_gordan_tensor_product_sparse, filter_ir_out=output_irreps, irrep_normalization="norm")
        elif tp_type == "GTP-grid":
            # The grid heuristics here have been set to pass the equivariance error for FP32 with 1e-2 (fails with 1e-3)
            # Based on https://shtools.github.io/SHTOOLS/grid-formats.html
            return x, y, partial(gaunt_tensor_product_s2grid, filter_ir_out=output_irreps, res_beta=2*lmax + 1 if FLAGS.lmax_based_grid else 100, res_alpha=2*(2*lmax+1) if FLAGS.lmax_based_grid else 99,  p_val1=1, p_val2=1, s2grid_fft=False, quadrature="gausslegendre")
        elif tp_type == "GTP-fourier":
            return x, y, partial(gaunt_tensor_product_2D_fourier, filter_ir_out=output_irreps, res_theta=300, res_phi=300, convolution_type="direct") # Remember to try with fft as well"
        elif  tp_type == "Matrix-TP":
            return x, y, partial(matrix_tensor_product, irrep_normalization="norm")
    else:
        raise ValueError(f"{tp_type} not supported")

@flops_counter
def func_flops(func, *args):
    return func(*args)

def benchmark_per_lmax(lmax: int, irreps_type: str, tp_type: str, batch: int):
    x, y, TP = tp_initializer(lmax, tp_type, irreps_type)
    TP = jax.jit(TP)
    # Compiler run
    start = time.time()
    for _ in range(10):
        result = TP(x, y)
        result.array.block_until_ready()
    print(f"Compiling took {(time.time() - start):.3f} s")
    print(f"irreps_type {FLAGS.irreps_type} tensor_product_type {FLAGS.tensor_product_type} lmax {FLAGS.lmax} batch {FLAGS.batch} num_irreps {result.irreps.num_irreps}")
    if FLAGS.ncu_flag:
        func_flops(TP, x, y)
    elif FLAGS.dot_graph:
        from functools import partial
        from jaxlib import xla_client

        from graphviz import Source
        from IPython.display import Image

        def todotgraph(x):
            return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))

        lmax_base_grid = "lmaxgrid" if FLAGS.lmax_based_grid else "fixedgrid"
        with open(f"benchmarking/dots/TP_{FLAGS.irreps_type}_{FLAGS.tensor_product_type}_{FLAGS.lmax}_{FLAGS.batch}_{lmax_base_grid}_jax.dot", "w") as file:
            file.write(todotgraph(jax.jit(TP).lower(x, y).compile().as_text()))
    else:
        timings = []
        # from ctypes import cdll
        # libcudart = cdll.LoadLibrary("libcudart.so")
        # libcudart.cudaProfilerStart()
        for _ in range(TRIALS):
            start = time.time()
            result = TP(x, y)
            result.array.block_until_ready()
            timings.append(time.time() - start)

        # libcudart.cudaProfilerStop()
        # Taking average of the last 20 iterations
        timings = timings[:-20]
        avg_time = np.mean(timings)
        print(f"Walltime took {avg_time*1000:3f} ms")
        with open(FLAGS.walltime_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([lmax, irreps_type, tp_type, batch, avg_time, result.irreps.num_irreps])
            
def main():
    benchmark_per_lmax(
        FLAGS.lmax, FLAGS.irreps_type, FLAGS.tensor_product_type, FLAGS.batch
    )

if __name__ == "__main__":
    main()