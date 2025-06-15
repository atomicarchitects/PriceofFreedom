#!/bin/bash

mkdir -p experiments/benchmarking/profiles

for batch in 10000;do
    for tp_type in "GTP-fourier" "CGTP-dense" "GTP-grid" "CGTP-sparse" "Matrix-TP" ;do
            for lmax in 1 2 3 4 5 6 7 8 9 10;do
                echo "$tp_type $irreps_type $lmax $batch"
                python experiments/benchmarking/utils/run_profiler.py experiments/benchmarking/__main__.py experiments/benchmarking/profiles/nsight_${irreps_type}_${tp_type}_${lmax}_${batch}_jax.csv \
                --tensor_product_type=$tp_type \
                --irreps_type=$irreps_type \
                --lmax=$lmax \
                --batch=$batch \
                --ncu_flag=True \
            done
        done
    done
done

rm -rf /tmp/jax_cache

# Postprocess the profiles
python -m experiments.benchmarking.utils.nsight_postprocess --profiles_path="experiments/benchmarking/profiles/RTX" --output_path="experiments/benchmarking/csv/RTX_nsight_profiling.csv"