#/bin/bash

# CIL CONFIG
NOTE="test" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="ours"

K_COEFF="2"
TEMPERATURE="2"

TRANSFORM_ON_GPU="--transform_on_gpu"
N_WORKER=4
FUTURE_STEPS=2
EVAL_N_WORKER=4
EVAL_BATCH_SIZE=1000
USE_KORNIA="--use_kornia"
#USE_KORNIA=""
UNFREEZE_RATE=0.25
SEEDS="1"


DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
ONLINE_ITER=0.6

SIGMA=10
REPEAT=1
INIT_CLS=100
USE_AMP="--use_amp"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=200
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=1281167
    MODEL_NAME="resnet18" EVAL_PERIOD=2000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python main_new.py --mode $MODE --transform_on_worker \
    --dataset $DATASET --unfreeze_rate $UNFREEZE_RATE $USE_KORNIA --k_coeff $K_COEFF --temperature $TEMPERATURE \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $TRANSFORM_ON_GPU --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE
done
