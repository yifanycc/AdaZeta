#!/bin/bash

TASK=${TASK:-SST-2}
K=${K:-16}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-80000}
EVAL_STEP=${EVAL_STEP:-1000}
MODEL=${MODEL:-roberta-large}
SHRINK=${SHRINK:-0.5}
TRAINER=${TRAINER:-mezo} # choose from standard/zeta
RATE=${RATE:-1}
LR_SCH=${LR_SCH:-1}
DEVICE=${DEVICE:-0}
MODE=${MODE:-ft}
export CUDA_VISIBLE_DEVICES=$DEVICE

if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--num_prefix 5 --no_reparam --prefix_init_by_real_act"
    TYPE="prefix"
elif [ "$MODE" == "lora" ]; then
    TYPE="lora"
elif [ "$MODE" == "adapters" ]; then
    TYPE="adapters"
elif [ "$MODE" == "adazeta" ]; then
    TYPE="adazeta"
elif [ "$MODE" == "prompt" ]; then
    TYPE="prompt"
elif [ "$MODE" == "bitfit" ]; then
    TYPE="bitfit"
elif [ "$MODE" == "ft" ]; then
    TYPE="ft"
elif [ "$MODE" == "ia3" ]; then
    TYPE="ia3"
fi

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "Step: $STEP; Eval step: $EVAL_STEP"

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP-inc$RATE-sch$LR_SCH-mode$MODE
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=${TAG:-k${K}-${MODEL}-mezo-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP --weight_decay $WD --zero_order_eps $EPS \
    --tuning_type $TYPE --save_total_limit 1 --save_step 5000 \
    --zero_order_optim --lr_scheduler_type "constant" --optimizer "sgd" --efficient_zero_order --lr_scheduler_type $LR_SCH --inc_rate $RATE --trainer=$TRAINER \
    $@
