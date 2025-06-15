"""
Benchmarking Script Usage
========================

This script benchmarks various tensor product implementations using JAX and e3nn-jax.
It is designed to be run directly or via shell scripts for batch experiments on CPU or GPU.

Example command-line usage:
--------------------------
python -m benchmarking \
  --tensor_product_type=CGTP-dense \
  --irreps_type=MIMO \
  --lmax=2 \
  --batch=1 \
  --device=gpu \
  --walltime_file=benchmarking/csv/walltime.csv

Flags:
------
--device:            'gpu' or 'cpu' (default: 'gpu')
--lmax:              Maximum l value (int, default: 2)
--batch:             Batch size (int, default: 1)
--irreps_type:       'MIMO', 'SIMO', or 'SISO' (default: 'MIMO')
--tensor_product_type: 'CGTP-dense', 'CGTP-sparse', 'GTP-grid', 'GTP-fourier', 'Matrix-TP' (default: 'CGTP-dense')
--ncu_flag:          Enable Nsight Compute profiling (bool, default: False)
--dot_graph:         Dump DOT graph (bool, default: False)
--lmax_based_grid:   Use lmax-based grid (bool, default: False)
--backward:          Benchmark backward pass (bool, default: False)
--walltime_file:     Output CSV file for walltime results (default: benchmarking/csv/walltime.csv)

Batch usage:
------------
See shell scripts in 'shell/' directory, e.g.:
  shell/run_cpu_jax_timing.sh
  shell/run_gpu_jax_timing.sh

These scripts loop over parameter sweeps and aggregate results in CSV files.

"""
import csv
import os
import sys
import functools
import logging
from typing import Union, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Only set up paths if running as main
if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(path)

from freedom.tensor_products.functional import clebsch_gordan_tensor_product_dense, clebsch_gordan_tensor_product_sparse, gaunt_tensor_product_s2grid, gaunt_tensor_product_2D_fourier, vector_gaunt_tensor_product_s2grid, matrix_tensor_product
from freedom.tensor_products.vector_spherical_harmonics import VSHCoeffs
from benchmarking.utils.fast_flops import flops_counter

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import flax.linen as nn
import numpy as np
from functools import partial
import time
from absl import flags

def define_flags():
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
    flags.DEFINE_bool(
        "backward",
        False,
        "Flag for backward benchmarking"
    )
    flags.DEFINE_string(
        "walltime_file",
        os.path.join(os.path.dirname(__file__), "csv", "walltime.csv"),
        "Path to Walltime file"
    )

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

def tp_initializer(lmax, tp_type, irreps_type, batch, lmax_based_grid):
    input_irreps, output_irreps = irreps_mapper[irreps_type](lmax)
    x = e3nn.normal(input_irreps, jax.random.PRNGKey(0), (batch,))
    y = e3nn.normal(input_irreps, jax.random.PRNGKey(1), (batch,))
    if tp_type == "CGTP-dense":
        return x, y, partial(clebsch_gordan_tensor_product_dense, filter_ir_out=output_irreps, irrep_normalization="norm")
    elif tp_type == "CGTP-sparse":
        return x, y, partial(clebsch_gordan_tensor_product_sparse, filter_ir_out=output_irreps, irrep_normalization="norm")
    elif tp_type == "GTP-grid":
        return x, y, partial(gaunt_tensor_product_s2grid, filter_ir_out=output_irreps, res_beta=2*lmax + 1 if lmax_based_grid else 100, res_alpha=2*(2*lmax+1) if lmax_based_grid else 99,  p_val1=1, p_val2=1, s2grid_fft=False, quadrature="gausslegendre")
    elif tp_type == "GTP-fourier":
        return x, y, partial(gaunt_tensor_product_2D_fourier, filter_ir_out=output_irreps, res_theta=300, res_phi=300, convolution_type="direct")
    elif tp_type == "Matrix-TP":
        return x, y, partial(matrix_tensor_product, irrep_normalization="norm")
    else:
        raise ValueError(f"{tp_type} not supported")

@flops_counter
def func_flops(func, *args):
    return func(*args)

def benchmark_per_lmax(lmax: int, irreps_type: str, tp_type: str, batch: int, FLAGS):
    x, y, TP = tp_initializer(lmax, tp_type, irreps_type, batch, FLAGS.lmax_based_grid)
    WARMUP = 10
    TRIALS = 100
    if FLAGS.backward:
        _output = TP(x, y)
        target_output = e3nn.normal(_output.irreps, jax.random.PRNGKey(2),  (_output.shape[0],))
        def loss_fn(x, y, target_output):
            output = TP(x,y)
            return jnp.mean(jnp.square(output.array - target_output.array))
        loss_fn = jax.jit(jax.grad(loss_fn))
        # Compiler run
        for _ in range(WARMUP):
            result = loss_fn(x, y, target_output)
            result.array.block_until_ready()
        timings = []
        for _ in range(TRIALS):
            start = time.time()
            result = loss_fn(x, y, target_output)
            result.array.block_until_ready()
            timings.append(time.time() - start)
        timings = timings[:-20]
        avg_time = np.mean(timings)
        logging.info(f"Walltime took {avg_time*1000:3f} ms")
        try:
            os.makedirs(os.path.dirname(FLAGS.walltime_file), exist_ok=True)
            with open(FLAGS.walltime_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([lmax, irreps_type, tp_type, batch, avg_time, _output.irreps.num_irreps])
        except Exception as e:
            logging.error(f"Failed to write to {FLAGS.walltime_file}: {e}")
    else:
        TP = jax.jit(TP)
        # Compiler run
        for _ in range(WARMUP):
            result = TP(x, y)
            result.array.block_until_ready()
        logging.info(f"irreps_type {FLAGS.irreps_type} tensor_product_type {FLAGS.tensor_product_type} lmax {FLAGS.lmax} batch {FLAGS.batch} normalization {result.irreps.num_irreps}")
        if FLAGS.ncu_flag:
            func_flops(TP, x, y)
        elif FLAGS.dot_graph:
            from functools import partial
            from jaxlib import xla_client
            def todotgraph(x):
                return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))
            lmax_base_grid = "lmaxgrid" if FLAGS.lmax_based_grid else "fixedgrid"
            dot_path = os.path.join(os.path.dirname(__file__), f"dots/TP_{FLAGS.irreps_type}_{FLAGS.tensor_product_type}_{FLAGS.lmax}_{FLAGS.batch}_{lmax_base_grid}_jax.dot")
            os.makedirs(os.path.dirname(dot_path), exist_ok=True)
            with open(dot_path, "w") as file:
                file.write(todotgraph(jax.jit(TP).lower(x, y).compile().as_text()))
        else:
            timings = []
            for _ in range(TRIALS):
                start = time.time()
                result = TP(x, y)
                result.array.block_until_ready()
                timings.append(time.time() - start)
            timings = timings[:-20]
            avg_time = np.mean(timings)
            logging.info(f"Walltime took {avg_time*1000:3f} ms")
            try:
                os.makedirs(os.path.dirname(FLAGS.walltime_file), exist_ok=True)
                with open(FLAGS.walltime_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([lmax, irreps_type, tp_type, batch, avg_time, result.irreps.num_irreps])
            except Exception as e:
                logging.error(f"Failed to write to {FLAGS.walltime_file}: {e}")

def main():
    """Entry point for benchmarking. Parses flags, sets up JAX, and runs the benchmark."""
    define_flags()
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    # Set JAX compilation cache
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    # Set device globally
    try:
        jax.config.update("jax_platform_name", FLAGS.device)
        if jax.devices()[0].platform != FLAGS.device:
            logging.warning(f"Requested device {FLAGS.device} not available. Using {jax.devices()[0].platform} instead.")
        else:
            logging.info(f"Using device: {FLAGS.device}")
    except Exception as e:
        logging.error(f"Error setting device: {e}")
        sys.exit(1)
    benchmark_per_lmax(
        FLAGS.lmax, FLAGS.irreps_type, FLAGS.tensor_product_type, FLAGS.batch, FLAGS
    )

if __name__ == "__main__":
    main()