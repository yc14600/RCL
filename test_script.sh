
RUNS=5
for BENCHMARK in 'splitmnist' 'splitcifar10';
do
    
    RPL='replay'
    DVC='cuda:0'
    if [ "$BENCHMARK" = "splitmnist" ]; then
        EP=1
        MEM=200
        ZD=64
    elif [ "$BENCHMARK" = "splitcifar10" ]; then
        EP=10
        MEM=1000
        ZD=256
    fi
    echo $BENCHMARK
    for STP in 'naive' 'ewc' 'si' 'rwalk' 'agem' 'cwr' 'lwf' 'gdumb'; 
    do
        if [ "$STP" = "agem" ]; then
            RPL=''
        elif [ "$STP" = "gdumb" ]; then
            RPL='gdumb'
            STP='naive'
        fi
        SEED=0
        echo $STP
        while [ $SEED -lt $RUNS ]
        do

            echo $SEED
            python all_in_one_test.py -bmk $BENCHMARK -ep $EP -stype $STP -rpl $RPL -dvc $DVC -ms $MEM -zd $ZD
                    
            ((SEED++))
        done
    done
done
