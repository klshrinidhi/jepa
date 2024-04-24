#! /bin/bash

set -o errexit
set -o nounset
set -o xtrace

python -m app.main \
	--fname "configs/pretrain/vitb16.yaml" \
	--devices "cuda:0" \
	--log_wandb
