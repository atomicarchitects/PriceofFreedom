
#!/bin/bash

mkdir -p benchmarking/csv

csv_file="benchmarking/csv/walltime_gpu.csv"

echo "lmax,tp_type,irreps_type,batch,walltime" >> $csv_file

for batch in 1
do
    for tp_type in "CGTP-dense" "CGTP-sparse" "GTP-grid" "GTP-fourier"
    do
        for irreps_type in "SISO" "SIMO" "MIMO"
        do
            for lmax in 1 2 3 4 5 6 7 8
            do
                python -m benchmarking --tensor_product_type=$tp_type --irreps_type=$irreps_type --lmax=$lmax --batch=$batch --device=cpu --walltime_file=$csv_file
            done
        done
    done
done

rm -rf /tmp/jax_cache