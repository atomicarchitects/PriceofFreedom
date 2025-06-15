# Price of Freedom: Exploring Tradeoffs between Expressivity and Computational Efficiency in Equivariant Tensor Products

![Bunny](./misc/bunny.png)

Image Credit: [Song Kim]([https://songkim.me)

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

Checkout `Gaunt-Tensor-Product-S2Grid`

## Benchmarking experiments

### CPU

```bash
bash shell/run_cpu_timing.sh
```

### GPU

```bash
bash shell/run_gpu_timing.sh
```

[Nsight Compute](https://developer.nvidia.com/nsight-compute) installation needed. Make sure the GPU counters are [enabled](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

```
bash shell/run_nsight_profiling.sh
```

