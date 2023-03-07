#/bin/bash

# CIL CONFIG
NOTE="bic_tinyimagenet_iter_6_sigma10" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="bic"

K_COEFF="4"
TEMPERATURE="0.125"

TRANSFORM_ON_GPU="--transform_on_gpu"
N_WORKER=4
FUTURE_STEPS=4
EVAL_N_WORKER=4
EVAL_BATCH_SIZE=1000
#USE_KORNIA="--use_kornia"
USE_KORNIA=""
UNFREEZE_RATE=0.25
SEEDS="3"


DATASET="tinyimagenet" # cifar10, cifar100, tinyimagenet, imagenet
ONLINE_ITER=4.72817087105564
SIGMA=10
REPEAT=1
INIT_CLS=100
USE_AMP="--use_amp"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000
    N_SMP_CLS="9" K="3"
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000
    N_SMP_CLS="2" K="3"
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000
    N_SMP_CLS="3" K="3"
    CANDIDATE_SIZE=200 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=200
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=1281167
    N_SMP_CLS="3" K="3"
    CANDIDATE_SIZE=1000 VAL_SIZE=2
    MODEL_NAME="resnet18" EVAL_PERIOD=8000 F_PERIOD=200000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=1 nohup python main_new.py --mode $MODE \
    --dataset $DATASET --unfreeze_rate $UNFREEZE_RATE $USE_KORNIA --k_coeff $K_COEFF --temperature $TEMPERATURE \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --samples_per_task 20000 \
    --rnd_seed $RND_SEED --val_memory_size $VAL_SIZE \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $TRANSFORM_ON_GPU --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE &
done
