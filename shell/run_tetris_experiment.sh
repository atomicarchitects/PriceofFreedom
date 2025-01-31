#!/bin/bash

for lmax in 1 2 3 4; do
    python -m src --config configs/tetris/simple_network.py --tensor_product_config configs/tensor_products/clebsch_gordan_dense.py --config.hidden_lmax=${lmax} --wandb_tags=["tetris-camera-ready"] --workdir workdirs/tetris_experiment/clebsch_gordan_dense/hidden_lmax=${lmax}
    python -m src --config configs/tetris/simple_network.py --tensor_product_config configs/tensor_products/gaunt_s2grid.py         --config.hidden_lmax=${lmax} --wandb_tags=["tetris-camera-ready"] --workdir workdirs/tetris_experiment/gaunt_s2grid/hidden_lmax=${lmax}
done
