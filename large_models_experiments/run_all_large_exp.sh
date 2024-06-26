export WANDB_START_METHOD="thread"

#SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP



MODEL=meta-llama/Llama-2-7b-hf TASK=WIC MODE=loretta_adp LR=1e-4 RANK=5 EPS=1e-3 BS=16 DEVICE=0 NUM=20 NUM_INIT=1 SHRINK=0.85 LR_SCH=constant TRAINER=zeta bash mezo_local.sh &
MODEL=meta-llama/Llama-2-7b-hf TASK=CB MODE=loretta_adp LR=1e-4 RANK=5 EPS=1e-3 BS=16 DEVICE=1 NUM=20 NUM_INIT=1 SHRINK=0.85 LR_SCH=constant TRAINER=zeta bash mezo_local.sh &

wait
