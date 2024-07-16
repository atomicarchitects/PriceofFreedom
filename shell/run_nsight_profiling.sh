#!/bin/bash

mkdir -p benchmarking/profiles

for batch in 1
do
    for tp_type in "CGTP-dense" "CGTP-sparse" "GTP-grid" "GTP-fourier"
    do
        for irreps_type in "SIMO" "SISO" "MIMO"
        do
            for lmax in 1 2 3 4 5 6 7 8
            do
                echo "$tp_type $irreps_type $lmax $batch"
                python -m benchmarking.utils.run_profiler benchmarking.__main__.py benchmarking/profiles/batch_${1}/nsight_${irreps_type}_${tp_type}_${lmax}_${batch}.csv --tensor_product_type=$tp_type --irreps_type=$irreps_type --lmax=$lmax --batch=$batch --ncu_flag=True
            done
        done
    done
done

rm -rf /tmp/jax_cache
python -m benchmarking.utils.nsight_postprocess --profiles_path="benchmarking/profiles"