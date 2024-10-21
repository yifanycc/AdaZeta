MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
BLOCK=${BLOCK:-false}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-10000}
SHRINK=${SHRINK:-0.5}
EVAL_STEPS=${EVAL_STEPS:-200}
DEVICE=${DEVICE:-0}
NUM=${NUM:-15}
NUM_MAX=${NUM_MAX:-10}
RATE=${RATE:-0.4}
LR_SCH=${LR_SCH:-1}
TRAINER=${TRAINER:-mezo}
HEAD=${HEAD:-0} # 1 for tune/ 0 for no-tune
USE_NUM=${USE_NUM:-0} 1 for tru/ 0 for false
export CUDA_VISIBLE_DEVICES=$DEVICE
MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    TYPE="prefix"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
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
fi

TAG=mezo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$NUM-$RATE-$LR_SCH-$TRAINER-$HEAD-$NUM-$NUM_INIT-$USE_NUM-$HEAD-$SHRINK


TASK_ARGS=""
HEAD_TUNE=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD)
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP)
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

if [ "$HEAD" -eq 1 ]
then
   HEAD_TUNE="--tune_head"
else
  HEAD_TUNE=""
fi

if [ "$USE_NUM" -eq 1 ]
then
   USE_NUM="--use_num"
else
  USE_NUM=""
fi

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir ./result/$TASK-${MODEL_NAME}-$RANK-$TAG --tag $TAG --tuning_type $TYPE --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS --num_pertub $NUM \
    --trainer $TRAINER --rank $RANK --inc_rate $RATE \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type $LR_SCH --weight_decay=0.01\
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS --shrink_factor $SHRINK \
    --train_as_classification --num_pertub_max $NUM_MAX $USE_NUM --local_server\
    $EXTRA_ARGS \
    $TASK_ARGS \
    $HEAD_TUNE \
    "$@"
