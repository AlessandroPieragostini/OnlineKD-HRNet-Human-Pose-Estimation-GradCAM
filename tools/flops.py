from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config
from utils.utils import create_logger
from ptflop import get_model_complexity_info

import models


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


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Model
    model = eval('models.'+cfg.MODEL.NAME+'_crop.get_pose_net')(cfg, is_train=True)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    num_channels = 3
    macs, params, model_repr = get_model_complexity_info(
        model, (num_channels, 96, 128), as_strings=False, print_per_layer_stat=True, save_only=True, verbose=False)
    
    layers = model_repr.split('\n')
    print(layers[:2])
