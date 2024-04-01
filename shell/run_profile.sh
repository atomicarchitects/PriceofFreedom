#! /bin/bash

mkdir -p ../profiles

for LMAX in 3 4 5 6 7 8
do	
    for TPTYPE in "usual" "gaunt" "vectorgaunt"
	do	
		nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop -o ../profiles/${TPTYPE}_profile_l${LMAX} -f true python ../tetris.py --sh_lmax=$LMAX --hidden_lmax=$LMAX --tensor_product_type=$TPTYPE --profile=True
	done
done
