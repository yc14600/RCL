
RUNS=5
for BENCHMARK in 'splitmnist';
do
    
    
    DVC='cpu'
    if [ "$BENCHMARK" = "splitmnist" ]; then
        EP=1
        MEM=200
        ZD=64
        T=5
        C=10
    elif [ "$BENCHMARK" = "splitcifar10" ]; then
        EP=1
        MEM=2000
        ZD=256
        T=5
        C=10
    elif [ "$BENCHMARK" = "splitcifar100" ]; then
        EP=10
        MEM=5000
        ZD=512
        T=10
        C=100
    elif [ "$BENCHMARK" = "splitcifar110" ]; then
        EP=10
        MEM=5000
        ZD=512
        T=11
        C=110
    elif [ "$BENCHMARK" = "splittinyimagenet" ]; then
        EP=10
        MEM=10000
        ZD=1024
        T=10
        C=200
    fi
    echo $BENCHMARK
    for STP in 'gem'; #  'naive' 'lwf' 'gdumb' 'ewc' 'si' 'rwalk' 'cwr'  
    do
        if [ "$STP" = "agem" ]; then
            RPL="_"
        elif [ "$STP" = "gdumb" ]; then
            RPL='gdumb'
            STP='naive'
        else
            RPL='replay'
        fi
        SEED=4
        echo $STP
        while [ $SEED -lt $RUNS ]
        do

            echo $SEED
            python all_in_one_test.py -bmk $BENCHMARK -ep $EP -stype $STP -rpl $RPL -dvc $DVC -ms $MEM -zd $ZD -T $T -C $C -sd $SEED
                    
            ((SEED++))
        done
    done
done
