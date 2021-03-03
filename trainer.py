# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Modified by Sudeep Dasari
# --------------------------------------------------------
from __future__ import print_function

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    # TODO: Q2 check if model should be saved this epoch
    return epoch % args.save_freq == 0


def save_model(epoch, model_name, model):
    # TODO: Q2 Implement code for model saving
    path = f'{model_name}_{epoch:02d}.pt'
    torch.save(model.state_dict(), path)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO: Q1.5 Initialize your visualizer here!
    # TODO: Q1.2 complete your dataloader in voc_dataset.py
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    writer = SummaryWriter()

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    # TODO: Q1.4 Implement model training code!
    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            
            # TODO: your loss for multi-label clf?
            criterion = torch.nn.BCEWithLogitsLoss(wgt)
            loss = criterion(output, target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
            # Validation iteration
            writer.add_scalar('training loss', loss.item(), cnt)
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                writer.add_scalar('mean average precision', map.item(), cnt)
                model.train()
            cnt += 1
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)
            
        if scheduler is not None:
            scheduler.step()

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
