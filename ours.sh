#/bin/bash

# CIL CONFIG
NOTE="real_ours_cifar10_sigma0_class_balanced_balancing_retrieval_sample_weight" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="ours"
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=0
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
HUMAN_TRAINING="False"
USE_AMP="--use_amp"
SEEDS="1 2 3"
AVG_PROB="0.4"
RECENT_RATIO="0.8"
LOSS_BALANCING_OPTION="reverse_class_weight" #none
WEIGHT_METHOD="count_important"
WEIGHT_OPTION="loss"
USE_WEIGHT="classwise"
KLASS_WARMUP="300"
KLASS_TRAIN_WARMUP="50"
CURRICULUM_OPTION="class_acc"
VERSION="ver8"
INTERVAL=5
THRESHOLD="5e-2"
UNFREEZE_THRESHOLD="1e-1"
MAX_VALIDATION_INTERVAL=0
MIN_VALIDATION_INTERVAL=0
THRESHOLD_COEFF=0.1
THRESHOLD_POLICY="block"
UNFREEZE_COEFF=100

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000 ONLINE_ITER=1
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000 ONLINE_ITER=1
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000 ONLINE_ITER=0.75
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="resnet18" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=3 nohup python main.py --mode $MODE --loss_balancing_option $LOSS_BALANCING_OPTION \
    --dataset $DATASET --use_weight $USE_WEIGHT --klass_train_warmup $KLASS_TRAIN_WARMUP \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --weight_method $WEIGHT_METHOD --max_validation_interval $MAX_VALIDATION_INTERVAL \
    --rnd_seed $RND_SEED --weight_option $WEIGHT_OPTION --klass_warmup $KLASS_WARMUP --threshold $THRESHOLD --min_validation_interval $MIN_VALIDATION_INTERVAL \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME --version $VERSION --unfreeze_threshold $UNFREEZE_THRESHOLD \
    --lr $LR --batchsize $BATCHSIZE --recent_ratio $RECENT_RATIO --avg_prob $AVG_PROB --interval $INTERVAL --threshold_coeff $THRESHOLD_COEFF --unfreeze_coeff $UNFREEZE_COEFF \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --curriculum_option $CURRICULUM_OPTION --threshold_policy $THRESHOLD_POLICY \
    --note $NOTE --val_period $VAL_PERIOD --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP &
done
