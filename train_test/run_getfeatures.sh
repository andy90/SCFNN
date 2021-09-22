for((i=1001; i<1594; i++))
do
    cp -r G*parameters* D0/Config"$i"/
    cp produce_features_d_new.o D0/Config"$i"/
    cd D0/Config"$i"/
    ./produce_features_d_new.o
    cd ../../
done
