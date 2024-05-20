# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os
import sys
import time
import torch
import logging
import numpy as np
from tqdm import tqdm

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_pred_debug_images, save_gt_debug_images

logger = logging.getLogger(__name__)
np.set_printoptions(threshold=sys.maxsize)


def get_current_consistency_weight(current, rampup_length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def train(config, train_loader, model, criterion, criterion_kld, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    
    num_outputs = config.MODEL.N_STAGE if config.MODEL.MULTI else 1
    roles = [ 'Teacher_' if i == num_outputs-1 else f'Student{i+1}' 
                for i in range(num_outputs) ]
    loss_by_stage = [ AverageMeter() for i in range(num_outputs) ]
    acc_by_stage = [ AverageMeter() for i in range(num_outputs) ]

    losses = AverageMeter()
    losses_soft = AverageMeter()
    losses_hard = AverageMeter()
    losses_teacher = AverageMeter()

    pred_to_plot = []

    teacher_weight = config.TRAIN.TEACHER_WEIGHT
    kld_weight = config.TRAIN.KLD_WEIGHT
    cons_weight = get_current_consistency_weight(epoch, config.TRAIN.LENGTH)

    # Switch to train mode
    model.train()

    print("\nTraining...")
    with tqdm(train_loader, unit="batch", total=len(train_loader)) as tepoch:
        for i, (input, target, target_weight, meta) in enumerate(tepoch):    
            # compute outputs = outputs from stages
            outputs = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            
            loss_hard = 0
            loss_soft = 0
            teacher_loss = 0

            if isinstance(outputs, list):

                if not config.MODEL.MULTI:
                    outputs = [ outputs[-1] ]   
                else:
                    kld_couples = config.LOSS.KLD_COUPLES
                    dist_to = [ couple[0] for couple in kld_couples]

                for index, output in enumerate(outputs):                
                    if index == len(outputs) - 1:
                        teacher_loss = criterion(output, target, target_weight)
                        teacher_loss *= teacher_weight
                        stage_loss = teacher_loss
                    else:
                        ls = 0
                        ls_prova = 0
                        lh = 0
                        
                        if index+1 in dist_to and config.LOSS.USE_MSE:
                            lh += criterion(output, target, target_weight)
                        
                        if config.LOSS.USE_KLD:
                            for index_dist_from in range(index+1, config.MODEL.N_STAGE):
                                if [index+1, index_dist_from+1] in kld_couples:
                                    ls += criterion_kld(output, outputs[index_dist_from], target_weight)
                    
                        ls *= kld_weight * cons_weight
                        ls_prova *= kld_weight * cons_weight

                        loss_hard += lh
                        loss_soft += ls
                        stage_loss = lh + ls
                        
                    loss_by_stage[index].update(stage_loss, input.size(0))
                    _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
                    pred_to_plot.append(pred)
                    acc_by_stage[index].update(avg_acc, cnt)
            else:
                raise ValueError("Model output is not a list")

            loss = loss_hard + teacher_loss + loss_soft

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), input.size(0))
            losses_hard.update(loss_hard, input.size(0))
            losses_soft.update(loss_soft, input.size(0))
            losses_teacher.update(teacher_loss, input.size(0))

            to_print = f"Epoch {epoch+1} | Losses [{losses.avg*1000:.4f}, T {losses_teacher.avg*1000:.4f}, S {losses_soft.avg*1000:.4f}] PCK@0.5 ["
            for acc in acc_by_stage:
                to_print += f"{acc.avg:.4f}, "
            tepoch.set_description(to_print[:-2]+']')
                # f"Epoch {epoch+1} | Losses [{losses.avg*1000:.4f}, T {losses_teacher.avg*1000:.4f}, S {losses_soft.avg*1000:.4f}] "
                # f"PCK@0.50 [T {acc_by_stage[-1].avg:.4f}]")
            
            if i % config.PRINT_FREQ == 0:
                prefix = '{}_ep{}_b{}'.format(os.path.join(output_dir, 'train'), epoch, i)
                save_gt_debug_images(config, input, meta, target, prefix)
                for index in range(num_outputs):
                    save_pred_debug_images(config, input, meta, outputs[index], pred_to_plot[index]*4, prefix  + f"_{roles[index]}")

            metr_dict = {
                "loss": losses.avg,
                "teacher_loss": losses_teacher.avg.item(),
                "hard_loss": losses_hard.avg,
                "soft_loss": losses_soft.avg,
            }
            for i, acc in enumerate(acc_by_stage):
                metr_dict[f"acc_stage{i}"] = acc.avg.item()

    #once for epoch
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    '''
    for index in range(num_outputs):
        writer.add_scalar( roles[index] + '_train_loss', loss_by_stage[index].val, global_steps)
        writer.add_scalar( roles[index] + '_train_acc', acc_by_stage[index].val, global_steps)
    '''

    writer.add_scalar('train_loss', losses.val, global_steps)
    writer.add_scalar('train_loss_hard', losses_hard.val, global_steps)
    writer.add_scalar('train_loss_soft', losses_soft.val, global_steps)
    writer.add_scalar('train_loss_teacher', losses_teacher.val, global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    return metr_dict

def validate(config, val_loader, val_dataset, model, criterion, criterion_kld, output_dir,
             tb_log_dir, writer_dict=None, epoch = 0):
    
    num_outputs = config.MODEL.N_STAGE if config.MODEL.MULTI else 1
    roles = [ 'Teacher_' if i == num_outputs-1 else f'Student{i+1}' 
                for i in range(num_outputs) ]
    loss_by_stage = [ AverageMeter() for i in range(num_outputs) ]
    acc_by_stage = [ AverageMeter() for i in range(num_outputs) ]
    
    losses = AverageMeter()
    losses_soft = AverageMeter()
    losses_hard = AverageMeter()
    losses_teacher = AverageMeter()

    pred_to_plot = []

    # Switch to evaluate mode
    model.eval()
    
    num_samples = len(val_dataset)

    # N_STAGE corrisponde al numero di stage e output
    all_preds = np.zeros(
        (num_outputs, num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32)
    
    # N_STAGE corrisponde al numero di stage e output
    all_boxes = np.zeros((num_samples, 6))
    
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    teacher_weight = config.TRAIN.TEACHER_WEIGHT
    kld_weight = config.TRAIN.KLD_WEIGHT
    cons_weight = get_current_consistency_weight(epoch, config.TRAIN.LENGTH)

    print("\nValidation...")
    with torch.no_grad():
        with tqdm(val_loader, unit="batch", total=len(val_loader)) as tepoch:
            for i, (input, target, target_weight, meta) in enumerate(tepoch):    
                # compute outputs = teacher_out, stud_out
                outputs = model(input)
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                num_images = input.size(0)
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
                
                loss_hard = 0
                loss_soft = 0
                teacher_loss = 0

                if isinstance(outputs, list):

                    if not config.MODEL.MULTI:
                        outputs = [ outputs[-1] ]   
                        # print("inp", input.size())
                        # print("pred", outputs[-1][0][0].size(), np.sum(outputs[-1][0][0].detach().cpu().numpy()))
                        # print("target", target[0][0].size(), np.sum(target[0][0].detach().cpu().numpy()))   #FIXME: Remove
                        # print(target.detach().cpu().numpy())
                    else:
                        kld_couples = config.LOSS.KLD_COUPLES
                        dist_to = [l[0] for l in kld_couples]

                    for index, output in enumerate(outputs):                
                        if index == len(outputs) - 1:
                            teacher_loss = criterion(output, target, target_weight)
                            teacher_loss *= teacher_weight
                            stage_loss = teacher_loss
                        else:
                            ls = 0
                            lh = 0
                            
                            if index+1 in dist_to and config.LOSS.USE_MSE:
                                lh += criterion(output, target, target_weight)
                            
                            if config.LOSS.USE_KLD:
                                for index_dist_from in range(index+1, config.MODEL.N_STAGE):
                                    if [index+1, index_dist_from+1] in kld_couples:
                                        if index - index_dist_from > 1:
                                            ls += criterion_kld(output, outputs[index_dist_from], target_weight, t=5)
                                        else:
                                            ls += criterion_kld(output, outputs[index_dist_from], target_weight)
                        
                            loss_hard += lh
                            loss_soft += ls
                            stage_loss = lh + ls
                            
                        loss_by_stage[index].update(stage_loss, input.size(0))
                        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())
                        pred_to_plot.append(pred)
                        acc_by_stage[index].update(avg_acc, cnt)
                        preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
                        all_preds[index][idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                        all_preds[index][idx:idx + num_images, :, 2:3] = maxvals
                else:
                    raise ValueError("Model output is not a list")

                # ls *= kld_weight * cons_weight
                loss = loss_hard + teacher_loss + loss_soft * kld_weight * cons_weight
                losses.update(loss.item(), num_images)
                losses_hard.update(loss_hard, input.size(0))
                losses_soft.update(loss_soft, input.size(0))
                losses_teacher.update(teacher_loss, input.size(0))

                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score

                image_path.extend(meta['image'])
                idx += num_images

                to_print = f"Epoch {epoch+1} | Losses [{losses.avg*1000:.4f}, T {losses_teacher.avg*1000:.4f}, S {losses_soft.avg*1000:.4f}] PCK@0.5 ["
                for acc in acc_by_stage:
                    to_print += f"{acc.avg:.4f}, "
                tepoch.set_description(to_print[:-2]+']')
                    # f"Epoch {epoch+1} | Losses [{losses.avg*1000:.4f}, T {losses_teacher.avg*1000:.4f}, S {losses_soft.avg*1000:.4f}] "
                    # f"PCK@0.50 [T {acc_by_stage[-1].avg:.4f}]")
                if i % config.PRINT_FREQ == 0:
                    prefix = '{}_ep{}_b{}'.format(os.path.join(output_dir, 'val'), epoch, i)

                    save_gt_debug_images(config, input, meta, target, prefix)
                    
                    for index in range(num_outputs):
                        save_pred_debug_images(config, input, meta, outputs[index], pred_to_plot[index]*4, prefix  + f"_{roles[index]}")
                    
        #once for epoch
        model_name = config.MODEL.NAME
        perf_indicators = [0]*num_outputs
        prefix = '{}_ep{}'.format(os.path.join(output_dir, 'val'), epoch)
        
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        
        for index in range(num_outputs):
            
            name_values, perf_indicators[index] = val_dataset.evaluate(
                config, all_preds[index], output_dir, all_boxes, image_path,
                filenames, imgnums
            )

            print(roles[index], "\n")

            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, model_name)
            else:
                _print_name_value(name_values, model_name)

            writer.add_scalar( roles[index] + '_valid_loss', loss_by_stage[index].avg, global_steps)
            writer.add_scalar( roles[index] + '_valid_acc', acc_by_stage[index].avg, global_steps)

            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars( roles[index] + '_valid', dict(name_value), global_steps)
            else:
                writer.add_scalars( roles[index] + '_valid', dict(name_values), global_steps)

        writer.add_scalar('valid_loss', losses.val, global_steps)
        writer.add_scalar('valid_loss_hard', losses_hard.val, global_steps)
        writer.add_scalar('valid_loss_soft', losses_soft.val, global_steps)
        writer.add_scalar('valid_loss_teacher', losses_teacher.val, global_steps)
        writer_dict['valid_global_steps'] += 1

        metr_dict = {
                "loss": losses.avg,
                "teacher_loss": losses_teacher.avg.item(),
                "hard_loss": losses_hard.avg,
                "soft_loss": losses_soft.avg,
            }
        for i, acc in enumerate(acc_by_stage):
            metr_dict[f"acc_stage{i}"] = acc.avg.item()

        return perf_indicators, metr_dict


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    # logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name[:6] + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
