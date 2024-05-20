# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pprint
import shutil
import argparse

import torch
import torch.optim
from numpy import exp
import torch.utils.data
import torch.nn.parallel
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from codecarbon import track_emissions
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from core.function import train
from config import update_config
from core.function import validate
from core.loss import JointsMSELoss
from core.loss import JointsKLDLoss
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint

import models
import logging
from data_load import get_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


@track_emissions(country_iso_code = "ITA", offline = True)
def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Model
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data
    train_dataset, train_loader, valid_dataset, valid_loader = get_loader(cfg)

    # Loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    criterion_kld = JointsKLDLoss(cfg).cuda()

    # Hyperparameters
    num_outputs = cfg.MODEL.N_STAGE if cfg.MODEL.MULTI else 1 
    best_perf = [0] * num_outputs
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')
    patience = 12       
    waiting = 0

    if cfg.MODEL.MULTI:
        kld_couples = cfg.LOSS.KLD_COUPLES
        stage_to_eval = set ([l[0] - 1 for l in kld_couples] + [l[1] - 1 for l in kld_couples])

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        # logger.info("Loading checkpoint '{}'...".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            # checkpoint_file, checkpoint['epoch']))

    if cfg.TRAIN.LR_TYPE == "custom":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda = 
        lambda ep : 1 if ep <= decay_start else exp(-lr_factor * (1 + (ep - decay_start) // decay_step)),
        last_epoch = last_epoch)
    elif cfg.TRAIN.LR_TYPE == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda ep : 1, last_epoch=last_epoch)

    decay_start = cfg.TRAIN.DECAY_START
    decay_step = cfg.TRAIN.DECAY_STEP
    lr_factor = cfg.TRAIN.LR_FACTOR

    # Training loop
    writer_dict = {
    'writer': SummaryWriter(log_dir=tb_log_dir),
    'train_global_steps': 0,
    'valid_global_steps': 0}

    metr_dicts = []
    metr_dicts_val = []

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        metr_dict = train(cfg, train_loader, model, criterion, criterion_kld, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        perf_indicators, metr_dict_val = validate(
            cfg, valid_loader, valid_dataset, model, criterion, criterion_kld,
            final_output_dir, tb_log_dir, writer_dict, epoch
        )

        best_model = False
        if cfg.MODEL.MULTI:
            for index in stage_to_eval:
                pf = perf_indicators[index]
                if pf >= best_perf[index]:
                    best_perf[index] = pf
                    best_model = True
                    waiting = 0
        else:
            if perf_indicators[0] >= best_perf[0]:
                best_perf[0] = perf_indicators[0]
                best_model = True
                waiting = 0
        
        if not best_model:
            waiting += 1

        # logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicators[-1],
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

        if waiting > patience:
            print(f"Early stop at epoch {epoch}")
            break

        lr_scheduler.step()
        metr_dicts.append(metr_dict)
        metr_dicts_val.append(metr_dict_val)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('Saving final model state to {}...'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

    with open(
        os.path.join(final_output_dir, "training_metr.csv"), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, metr_dicts[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(metr_dicts)
    with open(
        os.path.join(final_output_dir, "validation_metr.csv"), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, metr_dicts_val[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(metr_dicts_val)
    
    # print(metr_dicts_val)
    # print(type(metr_dicts_val))
    # for key in metr_dicts_val[0].keys():
    #     print(key)
    #     temp = []
    #     for line in metr_dict_val:
    #         print(line)
    #         temp.append(line[key])
    #     plt.plot(temp)
    #     plt.title(key)
    #     plt.imsave(os.path.join(final_output_dir, f"{key}.png"))
    for key in metr_dicts_val[0].keys():
        temp = []
        for line in metr_dicts_val:
            temp.append(line[key])
        plt.plot(temp)
        plt.title(key)
        plt.savefig(os.path.join(final_output_dir, f"{key}.png"))
        plt.close()

if __name__ == '__main__':
    main()
