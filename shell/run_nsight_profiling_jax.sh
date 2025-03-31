#!/bin/bash

mkdir -p benchmarking/profiles


lmax_based_grid=true

# Set the grid type based on the condition
if $lmax_based_grid; then
    grid_type="lmaxgrid"
else
    grid_type="fixedgrid"
fi

for batch in 10000;do
    # for tp_type in "GTP-fourier" "CGTP-dense" "GTP-grid" "CGTP-sparse" "Matrix-TP" ;do
    for tp_type in "GTP-fourier" ;do
        for irreps_type in "MIMO";do
            for lmax in 1 2 3 4 5 6 7 8 9 10;do
                echo "$tp_type $irreps_type $lmax $batch"
                python benchmarking/utils/run_profiler.py benchmarking/__main__.py benchmarking/profiles/nsight_${irreps_type}_${tp_type}_${lmax}_${batch}_${grid_type}_jax.csv \
                --tensor_product_type=$tp_type \
                --irreps_type=$irreps_type \
                --lmax=$lmax \
                --batch=$batch \
                --ncu_flag=True \
                --lmax_based_grid=$lmax_based_grid
            done
        done
    done
done

rm -rf /tmp/jax_cache
python -m benchmarking.utils.nsight_postprocess --profiles_path="benchmarking/profiles/RTX" --output_path="benchmarking/csv/RTX_nsight_profiling.csv"