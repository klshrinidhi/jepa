#! /bin/bash

set -o errexit
set -o nounset
set -o xtrace

cd ~/jepa
git pull origin main

#################################################################################
## Single Node - Single or Multi GPU

# python -m app.main \
# 	--fname "configs/pretrain/vitb16.yaml" \
# 	--log_wandb

#################################################################################
## Multi Node Multi GPU

CONFIG="configs/pretrain/vitl16.yaml"
NODE_RANK=0
N_NODES=5
MASTER_ADDR='10.128.15.246' # 'localhost'
MASTER_PORT=4532

if [ ${NODE_RANK} != 0 ]; then
	rsync -azP ${MASTER_ADDR}:/data/output/jepa/ /data/output/jepa/
fi

python -m app.main \
	--fname ${CONFIG} \
	--log_wandb \
	--node_rank ${NODE_RANK} \
	--n_nodes ${N_NODES} \
	--master_addr ${MASTER_ADDR} \
	--master_port ${MASTER_PORT}

#################################################################################
