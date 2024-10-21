export WANDB_START_METHOD="thread"
# Available tasks
# SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

# AdaZeta
MODEL=meta-llama/Llama-2-7b-hf TASK=WIC MODE=adazeta LR=1e-4 RANK=5 EPS=1e-3 BS=16 DEVICE=0 NUM=20 NUM_MAX=15 SHRINK=0.85 LR_SCH=constant TRAINER=zeta bash mezo.sh
#MODEL=meta-llama/Llama-2-7b-hf TASK=CB MODE=adazeta LR=1e-4 RANK=5 EPS=1e-3 BS=16 DEVICE=1 NUM=20 NUM_MAX=15 SHRINK=0.85 LR_SCH=constant TRAINER=zeta bash mezo.sh
#
## MeZO
#MODEL=meta-llama/Llama-2-7b-hf TASK=WIC MODE=ft LR=5e-7 RANK=5 EPS=1e-3 BS=16 DEVICE=0 NUM=20 NUM_INIT=1 SHRINK=0.85 LR_SCH=constant TRAINER=mezo bash mezo.sh
#MODEL=meta-llama/Llama-2-7b-hf TASK=CB MODE=ft LR=5e-7 RANK=5 EPS=1e-3 BS=16 DEVICE=1 NUM=20 NUM_INIT=1 SHRINK=0.85 LR_SCH=constant TRAINER=mezo bash mezo.sh
#
## MeZO-LoRA
#MODEL=meta-llama/Llama-2-7b-hf TASK=WIC MODE=lora LR=1e-4 RANK=5 EPS=1e-3 BS=16 DEVICE=0 NUM=20 NUM_INIT=1 SHRINK=0.85 LR_SCH=constant TRAINER=mezo bash mezo.sh
#MODEL=meta-llama/Llama-2-7b-hf TASK=CB MODE=lora LR=1e-4 RANK=5 EPS=1e-3 BS=16 DEVICE=1 NUM=20 NUM_INIT=1 SHRINK=0.85 LR_SCH=constant TRAINER=mezo bash mezo.sh

wait
