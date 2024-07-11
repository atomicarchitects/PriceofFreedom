for tp_type in "GTP-fourier"
do
    for irreps_type in "SISO"
    do
        for lmax in 1 2 3
        do
            echo "$tp_type $irreps_type $lmax"
            python -m benchmark_tp --tensor_product_type=$tp_type --irreps_type=$irreps_type --lmax=$lmax
            python -m utils.run_profiler benchmark_tp.py nsight_profiles/nsight_${irreps_type}_${tp_type}_${lmax}.csv --tensor_product_type=$tp_type --irreps_type=$irreps_type --lmax=$lmax --ncu_flag=True
        done
    done
done

python -m utils.postprocess --profiles_path="nsight_profiles"
rm -rf /tmp/jax_cache