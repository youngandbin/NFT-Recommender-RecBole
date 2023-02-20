for DATASET in azuki bayc coolcats doodles meebits
do

    python main.py \
        --model NGCF \
        --dataset $DATASET \
        --config 'baseline' &

done
