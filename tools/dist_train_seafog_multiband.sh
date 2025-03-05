MODEL=$1
CONFIG=$2
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
TIME=$(date "+%Y%m%d-%H%M%S")
CONFIG_FILE="configs/${MODEL}/${CONFIG}.py" 
WORK_DIR="work_dirs/seafog_multiband/${CONFIG}/${TIME}"
# WORK_DIR="/aipt/CZX/mmsegmentation_czx/work_dirs/seafog_multiband/segnext_mscan-l_1xb4-adamw-40k_seafog_multiband9-600*600/20240730-143409" 
# 配合resume 
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config ${CONFIG_FILE} \
    --work-dir=${WORK_DIR} \
    --resume \
    --launcher pytorch ${@:4}
# bash tools/dist_train_seafog_multiband.sh segnext_multiband segnext_mscan-l_1xb4-adamw-40k_seafog_multiband3-600*600
# export CUDA_VISIBLE_DEVICES=0,1,2,3