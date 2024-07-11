for tp_type in "CGTP-dense"
do
    for irreps_type in "SISO"
    do
        for lmax in 1 2 3 4 5 6 7 8
        do
            echo "$tp_type $irreps_type $lmax"
            python -m utils.run_profiler benchmark_tp.py nsight_profiles/nsight_${irreps_type}_${tp_type}_${lmax}.csv --lstensor_product_type=$tp_type --irreps_type=$irreps_type --lmax=$lmax --ncu_flag=True
        done
    done
done

python -m utils.postprocess --profiles_path="nsight_profiles" --output_path="nsight_profiles/nsight_processed.csv"
