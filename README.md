# Price of Freedom: Exploring Tradeoffs between Expressivity and Computational Efficiency in Equivariant Tensor Products

Paper: https://openreview.net/forum?id=0HHidbjwcf

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
pip install -r requirements.in
```

## Tetris Experiments

```bash
bash shell/run_tetris_experiments.sh
```
## Benchmarking

### CPU timing

```bash
bash shell/run_cpu_timing.sh
```

### GPU timing

```bash
bash shell/run_gpu_timing.sh
```

### Nsight Compute (GPU-only)

[Nsight Compute](https://developer.nvidia.com/nsight-compute) installation needed. Make sure the GPU counters are [enabled](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

```
bash shell/run_nsight_profiling.sh
```

