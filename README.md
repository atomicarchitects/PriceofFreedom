# [ICML'25] Price of Freedom: Exploring Tradeoffs between Expressivity and Computational Efficiency in Equivariant Tensor Products

The official implementation of the [Price of Freedom: Exploring Tradeoffs between Expressivity and Computational Efficiency in Equivariant Tensor Products](https://arxiv.org/abs/2506.13523), published at ICML 2025.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
uv pip install -e .
```

Alternatively, you can install the dependencies manually:
```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.in
```

## Tetris Experiments

```bash
bash shell/run_tetris_experiments.sh
```

## 3BPA experiments

Checkout `experiments/Gaunt-Tensor-Product-S2Grid`. Refer to the [original code](https://github.com/lsj2408/Gaunt-Tensor-Product) for reproducing the experiments

### Plotting

`experiments/Gaunt-Tensor-Product-S2Grid/analyze_results.ipynb`

## Benchmarking experiments

### CPU

```bash
bash shell/run_cpu_timing.sh
```

### GPU

```bash
bash shell/run_gpu_timing.sh
```

[Nsight Compute](https://developer.nvidia.com/nsight-compute) installation needed. Make sure the GPU counters are [enabled](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

```
bash shell/run_nsight_profiling.sh
```

### Plotting

`notebooks/benchmarking_viz`
