#!/bin/bash

python -m tetris --tensor_product_type=clebsch-gordan --hidden_lmax=1
python -m tetris --tensor_product_type=clebsch-gordan --hidden_lmax=2
python -m tetris --tensor_product_type=clebsch-gordan --hidden_lmax=3
python -m tetris --tensor_product_type=clebsch-gordan --hidden_lmax=4


python -m tetris --tensor_product_type=gaunt --hidden_lmax=1
python -m tetris --tensor_product_type=gaunt --hidden_lmax=2
python -m tetris --tensor_product_type=gaunt --hidden_lmax=3
python -m tetris --tensor_product_type=gaunt --hidden_lmax=4

