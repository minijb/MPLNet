from torch import nn
import torch
from tools import AverageMeter,wandb_log_train
from optimzer import AdamW
from config import cfg
from loss import FocalLoss
from timm.scheduler.cosine_lr import CosineLRScheduler
import wandb


from anomalib.utils.metrics import AUPRO, AUROC
import logging
import torch.nn.functional as F
import numpy as np
import json
import os

_logger = logging.getLogger('train')

main_cfg = cfg['train']
pretrained_cfg = main_cfg['pretrain']
train_cfg = main_cfg['train']


def pretrained_train(model:nn.Module, trainloader,device, num_training_steps:int, 
               log_interval: int = 1, savedir: str = None):
    
    # something in train  --------------------------------------------------
    
    # loss
    smooth_loss = nn.SmoothL1Loss()
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr  = pretrained_cfg['lr'])
    
    # tracing ------------------------------------------------------
    
    smooth_loss_trace = AverageMeter()

    # train ----------------------------------------------------------
    
    model.train()
    optimizer.zero_grad()
    
    
    step = 0
    train_model = True

    
    while train_model:
    
        for inputs, _, _ in trainloader:
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            sm_loss = smooth_loss(outputs, inputs)
            
            sm_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # ----- update tracing
            
            smooth_loss_trace.update(sm_loss.item())
            

                
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                        'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '.format(
                        step+1, num_training_steps, 
                        loss       = smooth_loss_trace, 
                        lr         = optimizer.param_groups[0]['lr'])
                        )
            

            
            step = step + 1
            
            if step == num_training_steps:
                train_model = False
                break