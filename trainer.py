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
    
def make_gradient_histograms(step, model, writer):
    grad_dict = dict()
    params = model.named_parameters()
    with torch.no_grad():
        for i, named_param in enumerate(params):
            name, param = named_param
            if 'conv' in name:
                layer_name = name.split('.')[0]
                if layer_name not in grad_dict:
                    grad_dict[layer_name] = []
                grad_dict[layer_name].append(param.grad.flatten())
                
        for name, grad_list in grad_dict.items():
            grads = torch.cat(grad_list)
            writer.add_histogram(f'{name}/gradient_histogram', grads, step)
    return grad_dict


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO: Q1.5 Initialize your visualizer here!
    # TODO: Q1.2 complete your dataloader in voc_dataset.py
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval', args=args)
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', args=args)
    writer = SummaryWriter(model_name)

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
                make_gradient_histograms(cnt, model, writer)
            # Validation iteration
            writer.add_scalar('training loss', loss.item(), cnt)
            if cnt % args.val_every == 0:
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                writer.add_scalar('mean average precision', map.item(), cnt)
            cnt += 1
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning rate', lr, cnt)
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)
        if scheduler is not None:
            #print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            scheduler.step()

    # Validation iteration
    #test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    if args.save_at_end:
        save_model(epoch, model_name, model)
    return ap, map
