MODEL=$1
CONFIG=$2
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
TIME=$(date "+%Y%m%d-%H%M%S")
CONFIG_FILE="configs/${MODEL}/${CONFIG}.py" 
WORK_DIR="work_dirs/${CONFIG}/${TIME}"
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
    --launcher pytorch ${@:4}
# bash tools/dist_train.sh hrnet fcn_hr18_4xb4-80k_seafog-512x512
# export CUDA_VISIBLE_DEVICES=0,1,2,3