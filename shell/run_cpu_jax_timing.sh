
#!/bin/bash

mkdir -p benchmarking/csv

csv_file="benchmarking/csv/walltime_cpu_jax.csv"

echo "lmax,irreps_type,tensor_product_type,batch,Time,num_irreps" >> $csv_file

for batch in 10000;do
    for tp_type in "Matrix-TP" "GTP-fourier" "CGTP-dense" "GTP-grid" "VGTP-grid" "CGTP-sparse";do
        for irreps_type in "SISO" "SIMO" "MIMO";do
            for lmax in 8 9 10;do
                echo "$tp_type $irreps_type $lmax $batch"
                python -m benchmarking \
                --tensor_product_type=$tp_type \
                --irreps_type=$irreps_type \
                --lmax=$lmax \
                --batch=$batch \
                --device=cpu \
                --walltime_file=$csv_file \
                --lmax_based_grid=True
            done
        done
    done
done

rm -rf /tmp/jax_cache