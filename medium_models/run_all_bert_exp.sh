export WANDB_START_METHOD="thread"

# GLUE datasets supported in this code (change in $TASK)
# MNLI, SST2, COLA, QQP, QNLI, RTE, MRPC, STSB


# AdaZeta (w.o. adaptive query schedule)
MODEL=$MODEL_PATH TASK=MNLI MODE=adazeta EXTRA_TAG=lora BS=16 K=16 LR=1e-4 RATE=0.4 LR_SCH=constant TRAINER=mezo DEVICE=0 bash mezo.sh --apply_lora

# MeZO-LoRA
MODEL=$MODEL_PATH TASK=MNLI MODE=lora EXTRA_TAG=lora BS=16 K=16 LR=1e-4 RATE=0.4 LR_SCH=constant TRAINER=mezo DEVICE=0 bash mezo.sh --apply_lora


wait
