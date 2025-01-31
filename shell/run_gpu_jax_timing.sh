
#!/bin/bash

mkdir -p benchmarking/csv

csv_file="benchmarking/csv/walltime_gpu_jax.csv"

echo "lmax,irreps_type,tensor_product_type,batch,Time,num_irreps" >> $csv_file

for batch in 10000;do
    # for tp_type in "GTP-fourier" "CGTP-dense" "GTP-grid" "VGTP-grid" "CGTP-sparse" "Matrix-TP";do
    for tp_type in "GTP-fourier";do
        for irreps_type in "MIMO" "SIMO" "SISO";do
            for lmax in 1 2 3 4 5 6 7 8 9 10;do
                python -m benchmarking \
                --tensor_product_type=$tp_type \
                --irreps_type=$irreps_type \
                --lmax=$lmax \
                --batch=$batch \
                --walltime_file=$csv_file \
                --lmax_based_grid=True
            done
        done
    done
done

rm -rf /tmp/jax_cache
