# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import pathlib
import multiprocessing as mp

import pprint
import yaml
import wandb
import torch

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument('--log_wandb',action='store_true',default=False)
parser.add_argument('--node_rank',type=int,default=None)
parser.add_argument('--n_nodes',type=int,default=None)
parser.add_argument('--master_addr',type=str,default=None)
parser.add_argument('--master_port',type=str,default=None)


def process_main(local_rank, fname, rank, world_size, master_addr, master_port, log_wandb):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)
    if local_rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    
    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # Log config
    params['world_size'] = world_size
    params['rank'] = rank
    params['log_wandb'] = log_wandb
    if rank == 0:
        # pprint.PrettyPrinter(indent=4).pprint(params)
        pathlib.Path(params['logging']['folder']).mkdir(parents=True,exist_ok=True)
        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)
        if log_wandb:
            wandb.init(project='ClinicalMAE',
                       entity='cerc-pac',
                       config=params,
                       name=params['logging']['folder'].split('/')[-1])

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(master_addr,master_port,rank,world_size)
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the app with loaded config
    app_main(params['app'], args=params)


if __name__ == '__main__':
    args = parser.parse_args()
    args.devices = [f'cuda:{i}'
                    for i in range(torch.cuda.device_count())]
    num_gpus = len(args.devices)
    if num_gpus == 1:
        process_main(0, args.fname, num_gpus, args.devices, args.log_wandb)
    else:
        if args.node_rank is None:
            world_size = num_gpus
            beg_rank = 0
        else:
            world_size = args.n_nodes * num_gpus
            beg_rank = args.node_rank * num_gpus
            
        mp.set_start_method('spawn')
        for local_rank in range(num_gpus):
            rank = beg_rank + local_rank
            mp.Process(
                target=process_main,
                args=(local_rank, args.fname, rank, world_size, 
                      args.master_addr, args.master_port, args.log_wandb)
            ).start()
