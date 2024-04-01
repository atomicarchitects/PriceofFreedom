#! /bin/bash

for LMAX in 3 4 5 6 7 8
do	
    for TPTYPE in "usual" "gaunt" "vectorgaunt"
    do
        python ../tetris.py --sh_lmax=$LMAX --hidden_lmax=$LMAX --tensor_product_type=$TPTYPE
    done
done

