# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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
# parser.add_argument(
#     '--devices', type=str, nargs='+', default=['cuda:0'],
#     help='which devices to use on local machine')
parser.add_argument('--log_wandb',action='store_true',default=False)

def process_main(args):
# def process_main(rank, fname, world_size, devices, log_wandb):
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {args.fname}')

    # Load config
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # Log config
    params['rank'] = rank
    params['log_wandb'] = args.log_wandb
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        pathlib.Path(params['logging']['folder']).mkdir(exist_ok=True)
        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)
        if args.log_wandb:
            wandb.init(project='ClinicalMAE',
                       entity='cerc-pac',
                       config=params,
                       name=params['logging']['folder'].split('/')[-1])

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the app with loaded config
    app_main(params['app'], args=params)


if __name__ == '__main__':
    args = parser.parse_args()
    process_main(args)
    # args.devices = [f'cuda:{i}'
    #                 for i in range(torch.cuda.device_count())]
    # num_gpus = len(args.devices)
    # if num_gpus == 1:
    #     process_main(0,args.fname,num_gpus,args.devices, args.log_wandb)
    # else:
    #     mp.set_start_method('spawn')
    #     for rank in range(num_gpus):
    #         mp.Process(
    #             target=process_main,
    #             args=(rank, args.fname, num_gpus, args.devices, args.log_wandb)
    #         ).start()
